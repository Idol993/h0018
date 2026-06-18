import re
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


DATETIME_KEYWORDS = [
    'time', 'date', 'datetime', 'timestamp', 'created_at', 'updated_at',
    'published_at', 'recorded_at', '时间', '日期', '创建时间', '更新时间',
    '发布时间', '记录时间', 'create_time', 'update_time', 'pub_time'
]

DRIFT_RELATED_KEYWORDS = [
    'source', 'split', '来源', '渠道', 'domain', 'platform',
    'group', 'batch', '批次', '分区'
] + DATETIME_KEYWORDS


def get_diagnostic_columns(df: pd.DataFrame, text_col: str, label_col: str) -> List[str]:
    """获取所有诊断相关列：文本列、标签列、以及可能用作切分的列"""
    cols = [text_col, label_col]
    for col in df.columns:
        if col in cols:
            continue
        col_lower = str(col).lower()
        if any(kw.lower() in col_lower for kw in DRIFT_RELATED_KEYWORDS):
            cols.append(col)
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= min(20, len(df) * 0.5):
            cols.append(col)
            continue
        if is_datetime_col(df[col], col):
            cols.append(col)
    return cols


def is_datetime_col(series: pd.Series, col_name: str = '') -> bool:
    """判断一列是否是时间/日期类型"""
    name_lower = str(col_name).lower()
    has_keyword = any(kw.lower() in name_lower for kw in DATETIME_KEYWORDS)

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(20)
        if len(sample) < 3:
            return has_keyword
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.8:
                date_pattern = re.compile(
                    r'(\d{4}[-/年.]\d{1,2}[-/月.]\d{1,2})'
                    r'|(\d{1,2}[-/月.]\d{1,2}[-/年.]\d{2,4})'
                    r'|(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2})'
                    r'|(\d{10,13})',
                    re.IGNORECASE
                )
                pattern_match_ratio = sum(
                    1 for s in sample if date_pattern.search(s.strip())
                ) / len(sample)
                if has_keyword and valid_ratio >= 0.5:
                    return True
                if valid_ratio >= 0.9 and pattern_match_ratio >= 0.5:
                    return True
                return False
            return False
        except Exception:
            return has_keyword
    return has_keyword


def parse_datetime_safe(series: pd.Series) -> pd.Series:
    """安全地将一列转为datetime，失败返回NaT"""
    try:
        return pd.to_datetime(series, errors='coerce', utc=False)
    except Exception:
        return pd.Series([pd.NaT] * len(series))


def detect_label_issues(
    texts: List[str],
    labels: List[str],
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    检测标签噪声，返回 (问题样本DataFrame, 诊断元信息)
    元信息包含：是否降级、使用的折数、降级原因
    """
    meta = {
        'skipped': False,
        'degraded': False,
        'n_splits_used': n_splits,
        'reason': '',
        'method': 'cleanlab_cv'
    }

    n_samples = len(texts)
    if n_samples < 10:
        meta['skipped'] = True
        meta['reason'] = f'样本数太少({n_samples}<10)，跳过标签噪声检测'
        return pd.DataFrame(columns=[
            'index', 'text', 'current_label', 'suggested_label',
            'confidence', 'original_confidence', 'score_gap'
        ]), meta

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    n_classes = len(label_encoder.classes_)

    if n_classes < 2:
        meta['skipped'] = True
        meta['reason'] = f'类别数太少({n_classes})，无法进行标签噪声检测'
        return pd.DataFrame(columns=[
            'index', 'text', 'current_label', 'suggested_label',
            'confidence', 'original_confidence', 'score_gap'
        ]), meta

    label_counts = pd.Series(labels).value_counts()
    min_class_count = label_counts.min()

    actual_splits = n_splits
    if min_class_count < actual_splits:
        actual_splits = max(2, min_class_count)
        if actual_splits < 2:
            meta['skipped'] = True
            meta['reason'] = f'最小类别仅{min_class_count}个样本，无法进行分层交叉验证'
            return pd.DataFrame(columns=[
                'index', 'text', 'current_label', 'suggested_label',
                'confidence', 'original_confidence', 'score_gap'
            ]), meta
        meta['degraded'] = True
        meta['n_splits_used'] = actual_splits
        meta['reason'] = f'最小类别仅{min_class_count}个样本，折数自动从{n_splits}调整为{actual_splits}'

    meta['n_splits_used'] = actual_splits

    try:
        from cleanlab.filter import find_label_issues

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=min(5000, n_samples), ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state, C=1.0))
        ])

        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        pred_probs = cross_val_predict(pipeline, texts, y_encoded, cv=skf, method='predict_proba')

        issues_mask = find_label_issues(
            labels=y_encoded,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence'
        )

        if isinstance(issues_mask, np.ndarray) and issues_mask.dtype == bool:
            issue_indices = np.where(issues_mask)[0]
        else:
            issue_indices = np.array(issues_mask, dtype=int)

        results = []
        for idx in issue_indices:
            true_label = labels[idx]
            pred_idx = int(np.argmax(pred_probs[idx]))
            suggested_label = label_encoder.inverse_transform([pred_idx])[0]
            confidence = float(pred_probs[idx][pred_idx])
            original_confidence = float(pred_probs[idx][y_encoded[idx]])

            results.append({
                'index': int(idx),
                'text': texts[idx],
                'current_label': true_label,
                'suggested_label': suggested_label,
                'confidence': round(confidence, 4),
                'original_confidence': round(original_confidence, 4),
                'score_gap': round(confidence - original_confidence, 4)
            })

        results.sort(key=lambda x: x['score_gap'], reverse=True)
        return pd.DataFrame(results), meta

    except ImportError:
        meta['method'] = 'simple_cv'
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=min(5000, n_samples), ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
        ])

        skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=random_state)
        pred_probs = cross_val_predict(pipeline, texts, y_encoded, cv=skf, method='predict_proba')

        results = []
        for idx in range(len(texts)):
            true_label_idx = y_encoded[idx]
            pred_idx = int(np.argmax(pred_probs[idx]))
            if pred_idx != true_label_idx:
                confidence = float(pred_probs[idx][pred_idx])
                original_confidence = float(pred_probs[idx][true_label_idx])
                if confidence - original_confidence > 0.1:
                    results.append({
                        'index': int(idx),
                        'text': texts[idx],
                        'current_label': labels[idx],
                        'suggested_label': label_encoder.inverse_transform([pred_idx])[0],
                        'confidence': round(confidence, 4),
                        'original_confidence': round(original_confidence, 4),
                        'score_gap': round(confidence - original_confidence, 4)
                    })

        results.sort(key=lambda x: x['score_gap'], reverse=True)
        return pd.DataFrame(results), meta


def compute_gini_impurity(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    proportions = counts / total
    return 1.0 - np.sum(proportions ** 2)


def analyze_class_balance(labels: List[str]) -> Dict[str, Any]:
    df_labels = pd.Series(labels)
    counts = df_labels.value_counts()
    proportions = df_labels.value_counts(normalize=True)

    result = {
        'class_counts': counts.to_dict(),
        'class_proportions': {k: round(v, 4) for k, v in proportions.to_dict().items()},
        'num_classes': len(counts),
        'total_samples': len(labels),
        'gini_impurity': round(compute_gini_impurity(counts.values), 4),
        'max_count': int(counts.max()),
        'min_count': int(counts.min()),
        'max_min_ratio': round(counts.max() / counts.min(), 2) if counts.min() > 0 else float('inf'),
        'small_classes': {k: int(v) for k, v in counts.items() if v < 50},
        'tiny_classes': {k: int(v) for k, v in counts.items() if v < 5},
        'suggestions': []
    }

    if result['gini_impurity'] > 0.3:
        if result['max_min_ratio'] > 10:
            result['suggestions'].append(
                '类别极度不平衡，建议对多数类进行欠采样或对少数类进行过采样（如SMOTE、ADASYN）'
            )
        else:
            result['suggestions'].append(
                '类别存在一定不平衡，建议考虑使用类别权重（class_weight）或数据增强策略'
            )

    if len(result['tiny_classes']) > 0:
        tiny_cls = ', '.join(result['tiny_classes'].keys())
        result['suggestions'].append(
            f'⚠️ 以下类别样本极少 (<5): {tiny_cls}，标签噪声检测可能已降级或跳过，强烈建议补充数据'
        )
    elif len(result['small_classes']) > 0:
        small_cls = ', '.join(result['small_classes'].keys())
        result['suggestions'].append(
            f'以下类别样本数较少 (<50): {small_cls}，建议进行文本增强（同义词替换、回译等）'
        )

    return result


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0
    p = p / p_sum
    q = q / q_sum
    return jensenshannon(p, q) ** 2


def check_distribution_drift(
    texts_a: List[str],
    texts_b: List[str],
    label_a: str = 'split_A',
    label_b: str = 'split_B',
    split_col_name: Optional[str] = None
) -> Dict[str, Any]:
    len_a = [len(str(t)) for t in texts_a]
    len_b = [len(str(t)) for t in texts_b]

    max_len = max(max(len_a) if len_a else 0, max(len_b) if len_b else 0, 100)
    bins = np.linspace(0, max_len, 30)
    hist_a, _ = np.histogram(len_a, bins=bins, density=True) if len_a else (np.zeros(29), bins)
    hist_b, _ = np.histogram(len_b, bins=bins, density=True) if len_b else (np.zeros(29), bins)
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    js_len = jensen_shannon_divergence(hist_a, hist_b)

    try:
        all_texts = list(texts_a) + list(texts_b)
        if not all_texts:
            raise ValueError('no texts')
        vectorizer = TfidfVectorizer(max_features=min(1000, len(all_texts)))
        tfidf = vectorizer.fit_transform(all_texts)

        if len(texts_a) > 0 and len(texts_b) > 0:
            tfidf_a = tfidf[:len(texts_a)].mean(axis=0).A1
            tfidf_b = tfidf[len(texts_a):].mean(axis=0).A1
            tfidf_a = tfidf_a + 1e-10
            tfidf_b = tfidf_b + 1e-10
            js_tfidf = jensen_shannon_divergence(tfidf_a, tfidf_b)
        else:
            js_tfidf = 0.0
    except Exception:
        js_tfidf = 0.0

    warnings = []
    if js_len > 0.2:
        warnings.append(f'文本长度分布漂移较大 (JS={js_len:.4f})')
    if js_tfidf > 0.2:
        warnings.append(f'词频分布漂移较大 (JS={js_tfidf:.4f})')

    return {
        'split_label_a': label_a,
        'split_label_b': label_b,
        'split_col_name': split_col_name,
        'size_a': len(texts_a),
        'size_b': len(texts_b),
        'text_length_js': round(float(js_len), 4),
        'tfidf_js': round(float(js_tfidf), 4),
        'avg_len_a': round(float(np.mean(len_a)), 2) if len_a else 0,
        'avg_len_b': round(float(np.mean(len_b)), 2) if len_b else 0,
        'len_hist_bins': bins.tolist(),
        'len_hist_a': hist_a.tolist(),
        'len_hist_b': hist_b.tolist(),
        'warnings': warnings
    }


def compute_drift_by_split_col(
    df: pd.DataFrame,
    text_col: str,
    split_col: str,
    mode: str = 'pairwise',
    time_granularity: str = 'half',
    custom_time_ranges: Optional[List[Tuple[Any, Any]]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    根据指定列计算分布漂移

    mode:
      - 'pairwise': 两两比较所有不同取值
      - 'first_vs_rest': 第一个取值 vs 其他所有
      - 'auto': 若取值<=5用pairwise，否则first_vs_rest
      - 时间列: 根据time_granularity切分

    time_granularity (仅时间列有效):
      - 'half': 时间前段 vs 时间后段（默认）
      - 'month': 按月份对比
      - 'quarter': 按季度对比
      - 'custom': 自定义时间范围对比

    custom_time_ranges: [(start1, end1), (start2, end2)] 自定义时间范围对

    Returns:
        (drift_results, meta_info)
    """
    results = []
    meta = {
        'method': '',
        'is_time_split': False,
        'time_granularity': time_granularity,
        'total_rows': len(df),
        'valid_time_rows': 0,
        'invalid_time_rows': 0,
        'time_periods': []
    }
    if split_col not in df.columns:
        return results, meta

    series = df[split_col]

    if is_datetime_col(series, split_col):
        dt_series = parse_datetime_safe(series)
        valid_mask = dt_series.notna()
        n_total = len(df)
        n_valid = valid_mask.sum()
        n_invalid = n_total - n_valid
        meta['valid_time_rows'] = int(n_valid)
        meta['invalid_time_rows'] = int(n_invalid)

        if n_valid >= 10:
            meta['is_time_split'] = True
            valid_df = df[valid_mask].copy()
            valid_dt = dt_series[valid_mask]
            valid_df['__dt__'] = valid_dt

            if time_granularity == 'month':
                valid_df['__period__'] = valid_dt.dt.to_period('M').astype(str)
                periods = sorted(valid_df['__period__'].unique())
                meta['time_periods'] = periods
                meta['method'] = f'按时间列[{split_col}]月份对比 ({len(periods)}个月)'
                if len(periods) >= 2:
                    for i in range(len(periods)):
                        for j in range(i + 1, len(periods)):
                            p_a, p_b = periods[i], periods[j]
                            texts_a = valid_df[valid_df['__period__'] == p_a][text_col].astype(str).tolist()
                            texts_b = valid_df[valid_df['__period__'] == p_b][text_col].astype(str).tolist()
                            if len(texts_a) > 0 and len(texts_b) > 0:
                                drift = check_distribution_drift(
                                    texts_a, texts_b,
                                    label_a=f'{p_a}',
                                    label_b=f'{p_b}',
                                    split_col_name=split_col
                                )
                                results.append(drift)

            elif time_granularity == 'quarter':
                valid_df['__period__'] = valid_dt.dt.to_period('Q').astype(str)
                periods = sorted(valid_df['__period__'].unique())
                meta['time_periods'] = periods
                meta['method'] = f'按时间列[{split_col}]季度对比 ({len(periods)}个季度)'
                if len(periods) >= 2:
                    for i in range(len(periods)):
                        for j in range(i + 1, len(periods)):
                            p_a, p_b = periods[i], periods[j]
                            texts_a = valid_df[valid_df['__period__'] == p_a][text_col].astype(str).tolist()
                            texts_b = valid_df[valid_df['__period__'] == p_b][text_col].astype(str).tolist()
                            if len(texts_a) > 0 and len(texts_b) > 0:
                                drift = check_distribution_drift(
                                    texts_a, texts_b,
                                    label_a=f'{p_a}',
                                    label_b=f'{p_b}',
                                    split_col_name=split_col
                                )
                                results.append(drift)

            elif time_granularity == 'custom' and custom_time_ranges and len(custom_time_ranges) >= 2:
                meta['method'] = f'按时间列[{split_col}]自定义范围对比'
                range_labels = []
                for idx, (start, end) in enumerate(custom_time_ranges):
                    start_dt = pd.to_datetime(start)
                    end_dt = pd.to_datetime(end)
                    if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
                        end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                    mask = (valid_dt >= start_dt) & (valid_dt <= end_dt)
                    count = int(mask.sum())
                    start_str = start_dt.strftime('%Y-%m-%d')
                    end_str = pd.to_datetime(end).strftime('%Y-%m-%d')
                    label = f'范围{idx+1}: {start_str} ~ {end_str}'
                    range_labels.append((label, mask, count))
                    meta['time_periods'].append(label)

                for i in range(len(range_labels)):
                    for j in range(i + 1, len(range_labels)):
                        label_a, mask_a, cnt_a = range_labels[i]
                        label_b, mask_b, cnt_b = range_labels[j]
                        if cnt_a > 0 and cnt_b > 0:
                            texts_a = valid_df[mask_a][text_col].astype(str).tolist()
                            texts_b = valid_df[mask_b][text_col].astype(str).tolist()
                            drift = check_distribution_drift(
                                texts_a, texts_b,
                                label_a=label_a,
                                label_b=label_b,
                                split_col_name=split_col
                            )
                            results.append(drift)

            else:
                sorted_idx = np.argsort(valid_dt.values)
                n_valid_sorted = len(sorted_idx)
                mid = n_valid_sorted // 2
                first_half_idx = valid_df.index[sorted_idx[:mid]]
                second_half_idx = valid_df.index[sorted_idx[mid:]]
                if len(first_half_idx) > 0 and len(second_half_idx) > 0:
                    texts_a = valid_df.loc[first_half_idx, text_col].astype(str).tolist()
                    texts_b = valid_df.loc[second_half_idx, text_col].astype(str).tolist()
                    drift = check_distribution_drift(
                        texts_a, texts_b,
                        label_a='时间前段',
                        label_b='时间后段',
                        split_col_name=split_col
                    )
                    results.append(drift)
                    meta['method'] = f'按时间列[{split_col}]前后段切分'
                    meta['time_periods'] = ['时间前段', '时间后段']

            return results, meta

    series_clean = series.dropna()
    unique_vals = series_clean.unique().tolist()

    if len(unique_vals) < 2:
        return results, meta

    if mode == 'auto':
        mode = 'pairwise' if len(unique_vals) <= 5 else 'first_vs_rest'

    if mode == 'first_vs_rest':
        first_val = unique_vals[0]
        mask_first = df[split_col] == first_val
        texts_a = df.loc[mask_first, text_col].astype(str).tolist()
        texts_b = df.loc[~mask_first, text_col].astype(str).tolist()
        drift = check_distribution_drift(
            texts_a, texts_b,
            label_a=str(first_val),
            label_b='其他',
            split_col_name=split_col
        )
        results.append(drift)
        meta['method'] = f'按列[{split_col}]切分 (first_vs_rest)'

    else:
        for i in range(len(unique_vals)):
            for j in range(i + 1, len(unique_vals)):
                val_a = unique_vals[i]
                val_b = unique_vals[j]
                texts_a = df[df[split_col] == val_a][text_col].astype(str).tolist()
                texts_b = df[df[split_col] == val_b][text_col].astype(str).tolist()
                if len(texts_a) > 0 and len(texts_b) > 0:
                    drift = check_distribution_drift(
                        texts_a, texts_b,
                        label_a=str(val_a),
                        label_b=str(val_b),
                        split_col_name=split_col
                    )
                    results.append(drift)
        meta['method'] = f'按列[{split_col}]切分 (pairwise)'

    return results, meta


def check_text_quality(texts: List[str]) -> Dict[str, Any]:
    issues = {
        'empty_texts': [],
        'pure_numeric': [],
        'potential_gibberish': [],
        'duplicates': [],
        'too_short': []
    }

    seen = {}
    for idx, text in enumerate(texts):
        t = str(text) if text is not None else ''

        if not t or not t.strip():
            issues['empty_texts'].append({'index': idx, 'text': t})
            continue

        if len(t.strip()) < 3:
            issues['too_short'].append({'index': idx, 'text': t})

        if re.match(r'^[\d\s\W]+$', t.strip()):
            issues['pure_numeric'].append({'index': idx, 'text': t})

        if t in seen:
            issues['duplicates'].append({
                'index': idx,
                'text': t,
                'first_occurrence': seen[t]
            })
        else:
            seen[t] = idx

        gibberish_patterns = [
            r'(.)\1{5,}',
            r'^[!@#$%^&*()_+\-=\[\]{};\'\\:"|,<.>/?`~]+$',
            r'^[a-z]{1,3}[0-9]{1,3}[a-z]{0,3}$'
        ]
        for pat in gibberish_patterns:
            if re.search(pat, t, re.IGNORECASE):
                issues['potential_gibberish'].append({'index': idx, 'text': t})
                break

    return issues


def check_split_stratification(
    labels: List[str],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    total = train_size + val_size + test_size
    train_size_norm = train_size / total
    val_size_norm = val_size / total
    test_size_norm = test_size / total

    val_test_size = val_size_norm + test_size_norm
    val_relative = val_size_norm / val_test_size if val_test_size > 0 else 0

    try:
        train_indices, temp_indices = train_test_split(
            list(range(len(labels))),
            test_size=val_test_size,
            stratify=labels,
            random_state=random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_relative),
            stratify=[labels[i] for i in temp_indices],
            random_state=random_state
        )
        stratify_ok = True
    except ValueError:
        train_indices, temp_indices = train_test_split(
            list(range(len(labels))),
            test_size=val_test_size,
            random_state=random_state
        )
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=(1 - val_relative),
            random_state=random_state
        )
        stratify_ok = False

    original_counts = pd.Series(labels).value_counts(normalize=True).sort_index()

    splits = {
        'train': [labels[i] for i in train_indices],
        'val': [labels[i] for i in val_indices],
        'test': [labels[i] for i in test_indices]
    }

    stratification_report = {}
    all_warnings = []

    if not stratify_ok:
        all_warnings.append('⚠️ 部分类别样本太少，无法进行严格分层抽样，分布偏差可能较大')

    for split_name, split_labels in splits.items():
        split_counts = pd.Series(split_labels).value_counts(normalize=True).sort_index()
        diffs = {}
        warnings = []

        for cls in original_counts.index:
            orig_p = original_counts.get(cls, 0)
            split_p = split_counts.get(cls, 0)
            diff = abs(split_p - orig_p)
            diffs[cls] = round(diff, 4)
            if diff > 0.05:
                warnings.append(
                    f'类别 {cls}: 原始占比 {orig_p:.2%}, {split_name}占比 {split_p:.2%}, 偏差 {diff:.2%}'
                )

        max_diff = max(diffs.values()) if diffs else 0
        stratification_report[split_name] = {
            'size': len(split_labels),
            'proportions': {k: round(v, 4) for k, v in split_counts.to_dict().items()},
            'max_deviation': round(max_diff, 4),
            'warnings': warnings
        }
        all_warnings.extend(warnings)

    return {
        'split_ratios': {'train': round(train_size_norm, 2), 'val': round(val_size_norm, 2), 'test': round(test_size_norm, 2)},
        'split_indices': {'train': train_indices, 'val': val_indices, 'test': test_indices},
        'stratification': stratification_report,
        'warnings': all_warnings
    }


def generate_augmentation_suggestions(class_balance: Dict[str, Any]) -> Dict[str, Any]:
    suggestions = {
        'needs_augmentation': len(class_balance['small_classes']) > 0,
        'small_classes': class_balance['small_classes'],
        'tiny_classes': class_balance.get('tiny_classes', {}),
        'strategies': []
    }

    if suggestions['needs_augmentation']:
        suggestions['strategies'] = [
            '同义词替换 (Synonym Replacement): 使用WordNet或同义词词典替换句子中的词',
            '回译 (Back-Translation): 将文本翻译成其他语言再翻译回来',
            '随机插入/删除/交换: 随机插入同义词、删除词或交换词序',
            '上下文增强: 使用MLM模型（如BERT）进行上下文感知的词替换',
            'EDA (Easy Data Augmentation): 综合上述简单策略'
        ]

    return suggestions


def dataframe_fingerprint(df: pd.DataFrame, text_col: str, label_col: str) -> str:
    """计算数据指纹，用于缓存判断。
    包含所有诊断相关列的多行全内容采样，确保任何位置的任何变化都能检测到。
    不对文本做截断，不遗漏长文本后半部分的变化。
    """
    n = len(df)
    if n == 0:
        return hashlib.md5(f"empty_{sorted(df.columns.tolist())}".encode('utf-8')).hexdigest()

    diagnostic_cols = get_diagnostic_columns(df, text_col, label_col)

    sample_size = min(200, n)
    if n <= sample_size:
        sample_indices = list(range(n))
    else:
        sample_indices = []
        step = n / sample_size
        for i in range(sample_size):
            idx = int(i * step)
            if idx >= n:
                idx = n - 1
            sample_indices.append(idx)
        if 0 not in sample_indices:
            sample_indices[0] = 0
        if (n - 1) not in sample_indices:
            sample_indices[-1] = n - 1

    hasher = hashlib.md5()
    hasher.update(f"{n}".encode('utf-8'))
    hasher.update(f"[{','.join(sorted(diagnostic_cols))}]".encode('utf-8'))

    for col in diagnostic_cols:
        if col not in df.columns:
            hasher.update(b'__MISSING_COL__')
            continue
        series = df[col]
        nunique = series.nunique()
        hasher.update(f"|{col}:{nunique}".encode('utf-8'))

    for idx in sample_indices:
        row_parts = []
        for col in diagnostic_cols:
            if col in df.columns:
                v = df.iloc[idx][col]
                if pd.isna(v):
                    row_parts.append('__NA__')
                else:
                    row_parts.append(str(v))
            else:
                row_parts.append('__MISSING__')
        row_str = f"{idx}:{'||'.join(row_parts)}"
        hasher.update(row_str.encode('utf-8'))

    return hasher.hexdigest()


def run_full_diagnostics(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_col: Optional[str] = None,
    drift_mode: str = 'auto',
    time_granularity: str = 'half',
    custom_time_ranges: Optional[List[Tuple[Any, Any]]] = None
) -> Dict[str, Any]:
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    label_issues, label_issues_meta = detect_label_issues(texts, labels)
    class_balance = analyze_class_balance(labels)
    text_quality = check_text_quality(texts)
    split_stratification = check_split_stratification(
        labels, train_ratio, val_ratio, test_ratio
    )

    drift_results = []
    drift_meta = {
        'split_col': split_col,
        'mode': drift_mode,
        'method': '',
        'is_time_split': False,
        'time_granularity': time_granularity
    }

    if split_col and split_col in df.columns and split_col != text_col and split_col != label_col:
        drift_results, extra_meta = compute_drift_by_split_col(
            df, text_col, split_col,
            mode=drift_mode,
            time_granularity=time_granularity,
            custom_time_ranges=custom_time_ranges
        )
        drift_meta.update(extra_meta)
    else:
        mid = len(texts) // 2
        if mid > 0 and len(texts) >= 20:
            drift = check_distribution_drift(
                texts[:mid], texts[mid:],
                label_a='前半部分', label_b='后半部分',
                split_col_name='(默认前后切分)'
            )
            drift_results.append(drift)
            drift_meta['method'] = '默认前后切分(前半vs后半)'
        else:
            drift_meta['method'] = '样本太少，未执行漂移检测'

    augmentation_suggestions = generate_augmentation_suggestions(class_balance)

    total_samples = len(df)
    noise_count = len(label_issues)

    if label_issues_meta['skipped']:
        cleanlab_score = None
        noise_ratio = None
    else:
        cleanlab_score = round(max(0, 1.0 - (noise_count / max(total_samples, 1))), 4)
        noise_ratio = round(noise_count / max(total_samples, 1), 4)

    balance_score = round(max(0, 1.0 - class_balance['gini_impurity']), 4)

    quality_scores = {
        'cleanlab_score': cleanlab_score,
        'noise_ratio': noise_ratio,
        'balance_score': balance_score,
        'quality_score': 0.0,
        'label_issues_meta': label_issues_meta
    }

    drift_penalty = 0.0
    for d in drift_results:
        if d['text_length_js'] > 0.2:
            drift_penalty += 0.05
        if d['tfidf_js'] > 0.2:
            drift_penalty += 0.1

    text_issue_count = sum(len(v) for v in text_quality.values())
    text_penalty = min(0.2, text_issue_count / max(total_samples, 1))

    cleanlab_component = cleanlab_score if cleanlab_score is not None else 0.5
    quality_score = (
        cleanlab_component * 0.4 +
        balance_score * 0.3 +
        (1 - drift_penalty) * 0.15 +
        (1 - text_penalty) * 0.15
    )
    quality_scores['quality_score'] = round(max(0, min(1, quality_score)), 4)

    return {
        'label_issues': label_issues,
        'label_issues_meta': label_issues_meta,
        'class_balance': class_balance,
        'text_quality': text_quality,
        'split_stratification': split_stratification,
        'distribution_drift': drift_results,
        'drift_meta': drift_meta,
        'augmentation_suggestions': augmentation_suggestions,
        'quality_scores': quality_scores,
        'total_samples': total_samples
    }
