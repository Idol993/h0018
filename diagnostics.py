import re
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


def detect_label_issues(
    texts: List[str],
    labels: List[str],
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    try:
        from cleanlab.filter import find_label_issues

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
        ])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
        return pd.DataFrame(results)

    except ImportError:
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000, random_state=random_state))
        ])

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
        return pd.DataFrame(results)


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

    if len(result['small_classes']) > 0:
        small_cls = ', '.join(result['small_classes'].keys())
        result['suggestions'].append(
            f'以下类别样本数较少 (<50): {small_cls}，建议进行文本增强（同义词替换、回译等）'
        )

    return result


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return jensenshannon(p, q) ** 2


def check_distribution_drift(
    texts_a: List[str],
    texts_b: List[str],
    label_a: str = 'split_A',
    label_b: str = 'split_B'
) -> Dict[str, Any]:
    len_a = [len(str(t)) for t in texts_a]
    len_b = [len(str(t)) for t in texts_b]

    bins = np.linspace(0, max(max(len_a), max(len_b), 100), 30)
    hist_a, _ = np.histogram(len_a, bins=bins, density=True)
    hist_b, _ = np.histogram(len_b, bins=bins, density=True)
    hist_a = hist_a + 1e-10
    hist_b = hist_b + 1e-10
    js_len = jensen_shannon_divergence(hist_a, hist_b)

    try:
        all_texts = list(texts_a) + list(texts_b)
        vectorizer = TfidfVectorizer(max_features=1000)
        tfidf = vectorizer.fit_transform(all_texts)

        tfidf_a = tfidf[:len(texts_a)].mean(axis=0).A1
        tfidf_b = tfidf[len(texts_a):].mean(axis=0).A1
        tfidf_a = tfidf_a + 1e-10
        tfidf_b = tfidf_b + 1e-10
        js_tfidf = jensen_shannon_divergence(tfidf_a, tfidf_b)
    except Exception:
        js_tfidf = 0.0

    return {
        'split_label_a': label_a,
        'split_label_b': label_b,
        'text_length_js': round(float(js_len), 4),
        'tfidf_js': round(float(js_tfidf), 4),
        'avg_len_a': round(float(np.mean(len_a)), 2),
        'avg_len_b': round(float(np.mean(len_b)), 2),
        'len_hist_bins': bins.tolist(),
        'len_hist_a': hist_a.tolist(),
        'len_hist_b': hist_b.tolist(),
        'warnings': []
    }


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

    original_counts = pd.Series(labels).value_counts(normalize=True).sort_index()

    splits = {
        'train': [labels[i] for i in train_indices],
        'val': [labels[i] for i in val_indices],
        'test': [labels[i] for i in test_indices]
    }

    stratification_report = {}
    all_warnings = []

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


def run_full_diagnostics(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    split_col: Optional[str] = None
) -> Dict[str, Any]:
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()

    label_issues = detect_label_issues(texts, labels)
    class_balance = analyze_class_balance(labels)
    text_quality = check_text_quality(texts)
    split_stratification = check_split_stratification(
        labels, train_ratio, val_ratio, test_ratio
    )

    drift_results = []
    if split_col and split_col in df.columns:
        unique_vals = df[split_col].dropna().unique()
        if len(unique_vals) >= 2:
            val_a, val_b = unique_vals[0], unique_vals[1]
            texts_a = df[df[split_col] == val_a][text_col].astype(str).tolist()
            texts_b = df[df[split_col] == val_b][text_col].astype(str).tolist()
            drift = check_distribution_drift(texts_a, texts_b, str(val_a), str(val_b))
            drift_results.append(drift)
    else:
        mid = len(texts) // 2
        if mid > 0:
            drift = check_distribution_drift(texts[:mid], texts[mid:], '前半部分', '后半部分')
            drift_results.append(drift)

    augmentation_suggestions = generate_augmentation_suggestions(class_balance)

    total_samples = len(df)
    noise_count = len(label_issues)
    quality_scores = {
        'cleanlab_score': round(max(0, 1.0 - (noise_count / max(total_samples, 1))), 4),
        'noise_ratio': round(noise_count / max(total_samples, 1), 4),
        'balance_score': round(max(0, 1.0 - class_balance['gini_impurity']), 4),
        'quality_score': 0.0
    }

    drift_penalty = 0.0
    for d in drift_results:
        if d['text_length_js'] > 0.2:
            drift_penalty += 0.05
        if d['tfidf_js'] > 0.2:
            drift_penalty += 0.1

    text_issue_count = sum(len(v) for v in text_quality.values())
    text_penalty = min(0.2, text_issue_count / max(total_samples, 1))

    quality_scores['quality_score'] = round(max(0, min(1, (
        quality_scores['cleanlab_score'] * 0.4 +
        quality_scores['balance_score'] * 0.3 +
        (1 - drift_penalty) * 0.15 +
        (1 - text_penalty) * 0.15
    ))), 4)

    return {
        'label_issues': label_issues,
        'class_balance': class_balance,
        'text_quality': text_quality,
        'split_stratification': split_stratification,
        'distribution_drift': drift_results,
        'augmentation_suggestions': augmentation_suggestions,
        'quality_scores': quality_scores,
        'total_samples': total_samples
    }
