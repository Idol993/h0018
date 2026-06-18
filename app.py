import io
import json
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any

from diagnostics import run_full_diagnostics, check_split_stratification


st.set_page_config(
    page_title='标注数据质量检测工具',
    page_icon='🔍',
    layout='wide',
    initial_sidebar_state='expanded'
)


def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def load_json(uploaded_file) -> pd.DataFrame:
    try:
        content = uploaded_file.read()
        data = json.loads(content)
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ['data', 'rows', 'samples', 'items']:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])
    except Exception:
        uploaded_file.seek(0)
        return pd.read_json(uploaded_file, lines=True)


def load_huggingface(dataset_name: str, split: str = 'train') -> Optional[pd.DataFrame]:
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split=split)
        return ds.to_pandas()
    except Exception as e:
        st.error(f'加载HuggingFace数据集失败: {str(e)}')
        return None


def get_text_and_label_candidates(df: pd.DataFrame):
    text_candidates = []
    label_candidates = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        if dtype == 'object' or 'string' in dtype:
            text_candidates.append(col)
            if df[col].nunique() <= min(50, len(df) * 0.1) and df[col].nunique() >= 2:
                label_candidates.append(col)
    if not label_candidates:
        label_candidates = list(df.columns)
    return text_candidates, label_candidates


def highlight_rows(row):
    styles = [''] * len(row)
    if 'score_gap' in row.index and row['score_gap'] > 0.5:
        styles = ['background-color: #ffcccc'] * len(row)
    elif 'score_gap' in row.index and row['score_gap'] > 0.3:
        styles = ['background-color: #ffeecc'] * len(row)
    return styles


def generate_html_report(
    df: pd.DataFrame,
    diagnostics_results: Dict[str, Any],
    text_col: str,
    label_col: str
) -> str:
    cb = diagnostics_results['class_balance']
    qs = diagnostics_results['quality_scores']
    drifts = diagnostics_results['distribution_drift']
    li = diagnostics_results['label_issues']
    tq = diagnostics_results['text_quality']
    aug = diagnostics_results['augmentation_suggestions']
    ss = diagnostics_results['split_stratification']

    drift_info = ''
    for d in drifts:
        drift_info += f'''
        <tr>
            <td>{d['split_label_a']} vs {d['split_label_b']}</td>
            <td style="color: {'red' if d['text_length_js'] > 0.2 else 'green'}">{d['text_length_js']}</td>
            <td style="color: {'red' if d['tfidf_js'] > 0.2 else 'green'}">{d['tfidf_js']}</td>
        </tr>
        '''

    class_rows = ''
    for cls, cnt in cb['class_counts'].items():
        prop = cb['class_proportions'][cls]
        small_cls = 'background: #ffebee;' if cnt < 50 else ''
        class_rows += f'''
        <tr style="{small_cls}">
            <td>{cls}</td>
            <td>{cnt}</td>
            <td>{prop:.2%}</td>
        </tr>
        '''

    suggestions_html = ''
    all_suggestions = cb['suggestions'] + aug['strategies'] + ss['warnings']
    for d in drifts:
        if d['text_length_js'] > 0.2:
            all_suggestions.append(
                f"文本长度分布漂移过大 ({d['split_label_a']} vs {d['split_label_b']}): JS={d['text_length_js']}，建议检查数据来源一致性"
            )
        if d['tfidf_js'] > 0.2:
            all_suggestions.append(
                f"词频分布漂移过大 ({d['split_label_a']} vs {d['split_label_b']}): JS={d['tfidf_js']}，建议检查数据来源一致性"
            )

    for i, s in enumerate(all_suggestions[:20], 1):
        suggestions_html += f'<li>{s}</li>'

    if not suggestions_html:
        suggestions_html = '<li>暂无明显问题，数据集质量良好</li>'

    text_quality_summary = f'''
    <tr><td>空文本</td><td>{len(tq['empty_texts'])}</td></tr>
    <tr><td>过短文本 (&lt;3字符)</td><td>{len(tq['too_short'])}</td></tr>
    <tr><td>纯数字文本</td><td>{len(tq['pure_numeric'])}</td></tr>
    <tr><td>潜在乱码</td><td>{len(tq['potential_gibberish'])}</td></tr>
    <tr><td>重复样本</td><td>{len(tq['duplicates'])}</td></tr>
    '''

    overall_color = '#4caf50' if qs['quality_score'] >= 0.7 else ('#ff9800' if qs['quality_score'] >= 0.5 else '#f44336')

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>标注数据质量诊断报告</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; color: #333; }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #4f8bf9; padding-bottom: 10px; }}
        h2 {{ color: #2c3e50; margin-top: 30px; }}
        .score-box {{ display: inline-block; padding: 20px 40px; background: {overall_color}; color: white; border-radius: 8px; font-size: 32px; font-weight: bold; }}
        .score-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
        .score-card {{ background: #f5f7fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .score-label {{ color: #666; font-size: 14px; margin-bottom: 8px; }}
        .score-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #f0f4ff; color: #2c3e50; }}
        .warning {{ color: #e67e22; font-weight: bold; }}
        .danger {{ color: #e74c3c; font-weight: bold; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <h1>🔍 标注数据质量诊断报告</h1>
    <p><strong>数据列:</strong> 文本列="{text_col}", 标签列="{label_col}"</p>
    <p><strong>样本总数:</strong> {diagnostics_results['total_samples']}</p>

    <h2>📊 综合质量评分</h2>
    <div class="score-box">{qs['quality_score']:.1%}</div>
    <div class="score-grid">
        <div class="score-card">
            <div class="score-label">CleanLab标注质量</div>
            <div class="score-value">{qs['cleanlab_score']:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">噪声样本比例</div>
            <div class="score-value" class="{'danger' if qs['noise_ratio'] > 0.1 else ''}">{qs['noise_ratio']:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">类别平衡得分</div>
            <div class="score-value">{qs['balance_score']:.1%}</div>
        </div>
        <div class="score-card">
            <div class="score-label">可疑标注数量</div>
            <div class="score-value" class="{'danger' if len(li) > 0 else ''}">{len(li)}</div>
        </div>
    </div>

    <h2>🏷️ 类别分布</h2>
    <p><strong>Gini不纯度:</strong> <span class="{'danger' if cb['gini_impurity'] > 0.3 else ''}">{cb['gini_impurity']}</span></p>
    <p><strong>最大/最小样本比:</strong> {cb['max_min_ratio']}x</p>
    <table>
        <tr><th>类别</th><th>样本数</th><th>占比</th></tr>
        {class_rows}
    </table>

    <h2>⚡ 分布漂移检测 (JS散度)</h2>
    <p style="color:#666;">阈值: 文本长度 JS>0.2 或 词频 JS>0.2 视为存在显著漂移</p>
    <table>
        <tr><th>切分方式</th><th>文本长度JS散度</th><th>TF-IDF词频JS散度</th></tr>
        {drift_info}
    </table>

    <h2>📝 文本质量检查</h2>
    <table>
        <tr><th>问题类型</th><th>数量</th></tr>
        {text_quality_summary}
    </table>

    <h2>💡 修复与优化建议</h2>
    <ul>{suggestions_html}</ul>
</body>
</html>'''
    return html


def get_download_link(html_content: str, filename: str = 'diagnostics_report.html') -> str:
    b64 = base64.b64encode(html_content.encode()).decode()
    return f'<a href="data:text/html;base64,{b64}" download="{filename}">📥 下载HTML诊断报告</a>'


def render_overview_tab(df: pd.DataFrame, diag: Dict[str, Any], text_col: str, label_col: str):
    col1, col2, col3, col4 = st.columns(4)
    qs = diag['quality_scores']

    with col1:
        st.metric('综合质量评分', f'{qs["quality_score"]:.1%}')
    with col2:
        st.metric('标注质量(CleanLab)', f'{qs["cleanlab_score"]:.1%}')
    with col3:
        st.metric('噪声样本比例', f'{qs["noise_ratio"]:.1%}', delta=f'{len(diag["label_issues"])}条')
    with col4:
        st.metric('类别平衡得分', f'{qs["balance_score"]:.1%}')

    st.markdown('---')

    cb = diag['class_balance']
    tq = diag['text_quality']
    drifts = diag['distribution_drift']

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader('📈 类别分布概览')
        classes = list(cb['class_counts'].keys())
        counts = list(cb['class_counts'].values())
        colors = px.colors.qualitative.Set3[:len(classes)]
        fig = go.Figure(data=[go.Pie(labels=classes, values=counts, hole=0.4, marker=dict(colors=colors))])
        fig.update_layout(height=400, title='各类别占比')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader('⚠️ 文本质量问题统计')
        issue_types = ['空文本', '过短文本', '纯数字', '潜在乱码', '重复样本']
        issue_counts = [
            len(tq['empty_texts']),
            len(tq['too_short']),
            len(tq['pure_numeric']),
            len(tq['potential_gibberish']),
            len(tq['duplicates'])
        ]
        fig = go.Figure([go.Bar(x=issue_types, y=issue_counts, marker_color='#4f8bf9')])
        fig.update_layout(height=400, yaxis_title='数量', title='文本质量问题分布')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📊 分布漂移检测')

    for d in drifts:
        st.markdown(f'**切分: {d["split_label_a"]} vs {d["split_label_b"]}**')
        c1, c2, c3 = st.columns(3)
        with c1:
            color_js_len = '🔴' if d['text_length_js'] > 0.2 else '🟢'
            st.metric(f'{color_js_len} 文本长度JS散度', f"{d['text_length_js']:.4f}")
        with c2:
            color_js_tf = '🔴' if d['tfidf_js'] > 0.2 else '🟢'
            st.metric(f'{color_js_tf} TF-IDF词频JS散度', f"{d['tfidf_js']:.4f}")
        with c3:
            st.metric('平均文本长度', f"{d['avg_len_a']:.0f} / {d['avg_len_b']:.0f}")

        bins = d['len_hist_bins']
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=centers, y=d['len_hist_a'], name=d['split_label_a'], fill='tozeroy'))
        fig.add_trace(go.Scatter(x=centers, y=d['len_hist_b'], name=d['split_label_b'], fill='tozeroy'))
        fig.update_layout(title='文本长度分布对比', height=300, xaxis_title='文本长度', yaxis_title='密度')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📄 一键导出诊断报告')
    html_report = generate_html_report(df, diag, text_col, label_col)
    st.markdown(get_download_link(html_report), unsafe_allow_html=True)


def render_label_noise_tab(df: pd.DataFrame, diag: Dict[str, Any], label_col: str):
    label_issues = diag['label_issues']

    if len(label_issues) == 0:
        st.success('🎉 未发现明显的标签噪声样本，标注质量看起来不错！')
        return

    st.markdown(f'### 🔍 发现 {len(label_issues)} 条可疑标注样本')
    st.info('💡 **score_gap** 表示模型建议标签与原标签的置信度差距，数值越大越可能是标注错误')

    sort_by = st.selectbox(
        '排序方式',
        ['可疑程度（降序）', '可疑程度（升序）', '原标签', '建议标签'],
        index=0
    )

    display_df = label_issues.copy()
    if sort_by == '可疑程度（降序）':
        display_df = display_df.sort_values('score_gap', ascending=False)
    elif sort_by == '可疑程度（升序）':
        display_df = display_df.sort_values('score_gap', ascending=True)
    elif sort_by == '原标签':
        display_df = display_df.sort_values('current_label')
    elif sort_by == '建议标签':
        display_df = display_df.sort_values('suggested_label')

    display_df['text'] = display_df['text'].apply(lambda x: x[:200] + '...' if len(str(x)) > 200 else x)

    styled = display_df.style.apply(highlight_rows, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.subheader('✏️ 批量修正标签')

    all_labels = sorted(df[label_col].unique().tolist())
    suggested_labels = sorted(label_issues['suggested_label'].unique().tolist())
    for sl in suggested_labels:
        if sl not in all_labels:
            all_labels.append(sl)

    selected_indices = st.multiselect(
        '选择要修正的样本（按数据集中的原始索引）',
        options=label_issues['index'].tolist(),
        format_func=lambda x: f"索引#{x}: {str(label_issues[label_issues['index'] == x]['text'].values[0])[:60]}..."
    )

    if selected_indices:
        col1, col2 = st.columns(2)
        with col1:
            first_idx = selected_indices[0]
            first_suggested = label_issues[label_issues['index'] == first_idx]['suggested_label'].values[0]
            suggested_default = all_labels.index(first_suggested) if first_suggested in all_labels else 0
            new_label = st.selectbox('新标签', options=all_labels, index=suggested_default)
        with col2:
            st.write('')
            apply_btn = st.button('✅ 应用修正并导出', type='primary')

        if apply_btn:
            modified_df = df.copy()
            modified_df.loc[selected_indices, label_col] = new_label
            st.success(f'已将 {len(selected_indices)} 条样本的标签修正为 "{new_label}"')

            csv = modified_df.to_csv(index=False)
            st.download_button(
                label='📥 下载修正后的CSV',
                data=csv,
                file_name='corrected_labels.csv',
                mime='text/csv'
            )

            st.session_state['modified_df'] = modified_df
            if st.button('🔄 基于修正后数据重新诊断'):
                st.session_state.pop('diagnostics', None)
                st.rerun()


def render_class_balance_tab(df: pd.DataFrame, diag: Dict[str, Any], text_col: str, label_col: str):
    cb = diag['class_balance']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('类别数量', cb['num_classes'])
    with col2:
        st.metric('总样本数', cb['total_samples'])
    with col3:
        color = '🔴' if cb['gini_impurity'] > 0.3 else '🟢'
        st.metric(f'{color} Gini不纯度', cb['gini_impurity'])
    with col4:
        ratio_display = '∞' if cb['max_min_ratio'] == float('inf') else f'{cb["max_min_ratio"]}x'
        color = '🔴' if cb['max_min_ratio'] > 10 else '🟢'
        st.metric(f'{color} 最大/最小样本比', ratio_display)

    st.markdown('---')
    st.subheader('📊 类别分布可视化')

    classes = list(cb['class_counts'].keys())
    counts = list(cb['class_counts'].values())
    props = [cb['class_proportions'][c] for c in classes]
    small_flags = [c in cb['small_classes'] for c in classes]

    fig = go.Figure()
    colors = ['#f44336' if s else '#4f8bf9' for s in small_flags]
    fig.add_trace(go.Bar(x=classes, y=counts, marker_color=colors, text=counts, textposition='auto'))
    fig.update_layout(
        title='各类别样本数量（红色表示样本数<50）',
        height=500,
        xaxis_tickangle=-45,
        yaxis_title='样本数'
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_pie = px.pie(
        names=classes, values=counts,
        title='各类别占比（点击图例可下钻查看）',
        hole=0.4
    )
    fig_pie.update_layout(height=500)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('---')
    st.subheader('📋 类别分布明细')

    detail_df = pd.DataFrame({
        '类别': classes,
        '样本数': counts,
        '占比': [f'{p:.2%}' for p in props],
        '是否小类别': ['🔴 是 (<50)' if s else '🟢 否' for s in small_flags]
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.subheader('💡 类别平衡建议')

    if cb['suggestions']:
        for i, s in enumerate(cb['suggestions'], 1):
            st.warning(f'**建议 {i}:** {s}')
    else:
        st.success('✅ 类别分布基本平衡，无需特别处理')

    aug = diag['augmentation_suggestions']
    if aug['needs_augmentation']:
        st.markdown('---')
        st.subheader('🔧 文本增强策略建议')
        st.info('以下类别样本数不足50，建议开启文本增强:')
        for cls, cnt in aug['small_classes'].items():
            st.markdown(f'- **{cls}**: {cnt} 条样本')

        st.markdown('**推荐增强策略:**')
        for i, s in enumerate(aug['strategies'], 1):
            st.markdown(f'{i}. {s}')


def render_distribution_drift_tab(df: pd.DataFrame, diag: Dict[str, Any], text_col: str, label_col: str):
    st.subheader('📊 切分策略检查')

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider('训练集比例', 0.5, 0.9, 0.7, 0.05)
    with col2:
        val_ratio = st.slider('验证集比例', 0.05, 0.3, 0.15, 0.05)
    with col3:
        test_ratio = st.slider('测试集比例', 0.05, 0.3, 0.15, 0.05)

    rerun_split = st.button('🔄 重新计算分层抽样')

    if rerun_split or 'split_result' not in st.session_state:
        split_result = check_split_stratification(
            df[label_col].astype(str).tolist(),
            train_size=train_ratio,
            val_size=val_ratio,
            test_size=test_ratio
        )
        st.session_state['split_result'] = split_result
    else:
        split_result = st.session_state['split_result']

    sr = split_result
    st.markdown(f'**切分比例:** 训练 {sr["split_ratios"]["train"]:.0%} / 验证 {sr["split_ratios"]["val"]:.0%} / 测试 {sr["split_ratios"]["test"]:.0%}')

    col_t, col_v, col_te = st.columns(3)
    with col_t:
        max_dev_t = sr['stratification']['train']['max_deviation']
        color = '🔴' if max_dev_t > 0.05 else '🟢'
        st.metric(f'{color} 训练集最大类别偏差', f'{max_dev_t:.2%}', sr['stratification']['train']['size'])
    with col_v:
        max_dev_v = sr['stratification']['val']['max_deviation']
        color = '🔴' if max_dev_v > 0.05 else '🟢'
        st.metric(f'{color} 验证集最大类别偏差', f'{max_dev_v:.2%}', sr['stratification']['val']['size'])
    with col_te:
        max_dev_te = sr['stratification']['test']['max_deviation']
        color = '🔴' if max_dev_te > 0.05 else '🟢'
        st.metric(f'{color} 测试集最大类别偏差', f'{max_dev_te:.2%}', sr['stratification']['test']['size'])

    if sr['warnings']:
        st.markdown('⚠️ **分层抽样偏差警告 (>5%):**')
        for w in sr['warnings']:
            st.warning(w)
    else:
        st.success('✅ 三集合类别分布与原始数据一致，分层抽样良好')

    st.markdown('---')
    st.subheader('📈 各集合类别分布对比')

    all_classes = sorted(df[label_col].unique().tolist())
    original_props = df[label_col].value_counts(normalize=True).reindex(all_classes).fillna(0).tolist()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=all_classes, y=[p * 100 for p in original_props], name='原始数据', marker_color='#2c3e50'))

    for split_name in ['train', 'val', 'test']:
        split_props = []
        split_info = sr['stratification'][split_name]['proportions']
        for cls in all_classes:
            split_props.append(split_info.get(cls, 0))
        fig.add_trace(go.Bar(x=all_classes, y=[p * 100 for p in split_props], name=split_name))

    fig.update_layout(
        barmode='group',
        title='各集合类别分布对比（%）',
        height=500,
        xaxis_tickangle=-45,
        yaxis_title='占比 (%)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('⚡ 分布漂移检测结果')

    for i, d in enumerate(diag['distribution_drift']):
        st.markdown(f'#### 切分 {i + 1}: {d["split_label_a"]} vs {d["split_label_b"]}')

        c1, c2 = st.columns(2)
        with c1:
            js_len = d['text_length_js']
            status_len = '🔴 显著漂移' if js_len > 0.2 else '🟢 正常'
            st.metric(f'文本长度 JS散度: {status_len}', f'{js_len:.4f}')
            if js_len > 0.2:
                st.warning(f'文本长度分布差异较大，两切分平均长度: {d["avg_len_a"]:.0f} vs {d["avg_len_b"]:.0f}')

        with c2:
            js_tf = d['tfidf_js']
            status_tf = '🔴 显著漂移' if js_tf > 0.2 else '🟢 正常'
            st.metric(f'TF-IDF词频 JS散度: {status_tf}', f'{js_tf:.4f}')
            if js_tf > 0.2:
                st.warning('词频分布差异较大，可能导致训练/测试分布不一致')

        bins = d['len_hist_bins']
        centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=centers, y=d['len_hist_a'],
            name=d['split_label_a'], fill='tozeroy', mode='lines'
        ))
        fig.add_trace(go.Scatter(
            x=centers, y=d['len_hist_b'],
            name=d['split_label_b'], fill='tozeroy', mode='lines'
        ))
        fig.update_layout(
            title='文本长度分布密度对比',
            height=350,
            xaxis_title='文本长度（字符数）',
            yaxis_title='密度'
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    st.title('🔍 标注数据质量检测与数据集诊断工具')
    st.markdown('专为NLP团队设计：检测标签噪声、分析类别不平衡、监控分布漂移、评估文本质量')

    if 'df' not in st.session_state:
        st.session_state['df'] = None
    if 'diagnostics' not in st.session_state:
        st.session_state['diagnostics'] = None
    if 'text_col' not in st.session_state:
        st.session_state['text_col'] = None
    if 'label_col' not in st.session_state:
        st.session_state['label_col'] = None
    if 'modified_df' not in st.session_state:
        st.session_state['modified_df'] = None

    with st.sidebar:
        st.header('📂 数据输入')

        input_mode = st.radio(
            '选择数据输入方式',
            ['上传本地文件', 'HuggingFace数据集', '使用示例数据'],
            index=0
        )

        df = None

        if input_mode == '上传本地文件':
            uploaded_file = st.file_uploader(
                '拖入或选择标注文件',
                type=['csv', 'json', 'jsonl'],
                help='支持CSV、JSON数组、JSON Lines格式'
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = load_csv(uploaded_file)
                    else:
                        df = load_json(uploaded_file)
                    st.success(f'✅ 成功加载 {len(df)} 条数据')
                except Exception as e:
                    st.error(f'文件加载失败: {str(e)}')

        elif input_mode == 'HuggingFace数据集':
            hf_name = st.text_input('HuggingFace数据集名称', placeholder='例如: imdb, ag_news')
            hf_split = st.text_input('数据集split', value='train')
            if st.button('📥 加载数据集'):
                if hf_name:
                    with st.spinner('正在从HuggingFace加载...'):
                        df = load_huggingface(hf_name, hf_split)
                        if df is not None:
                            st.success(f'✅ 成功加载 {len(df)} 条数据')

        elif input_mode == '使用示例数据':
            st.info('使用内置的示例数据（模拟分类标注）')
            if st.button('🔄 生成示例数据'):
                np.random.seed(42)
                n = 500
                labels = np.random.choice(['正面', '负面', '中性'], size=n, p=[0.5, 0.3, 0.2])
                texts = []
                for lbl in labels:
                    if lbl == '正面':
                        templates = ['这个产品非常好，很满意', '效果不错，推荐购买', '服务态度很好，点赞', '质量很棒，超出预期']
                    elif lbl == '负面':
                        templates = ['很差劲，不推荐', '质量有问题，退货了', '服务态度差，很失望', '完全不值这个价']
                    else:
                        templates = ['一般般，没有惊喜', '还可以吧，凑合用', '没什么特别的感觉', '不好不坏，普普通通']
                    t = templates[np.random.randint(len(templates))]
                    if np.random.random() < 0.08:
                        t = '' if np.random.random() < 0.3 else t + ' ' + ''.join(['a' * 10])
                    texts.append(t)

                labels_with_noise = labels.copy()
                noise_idx = np.random.choice(n, size=25, replace=False)
                for idx in noise_idx:
                    other_labels = [l for l in ['正面', '负面', '中性'] if l != labels[idx]]
                    labels_with_noise[idx] = np.random.choice(other_labels)

                df = pd.DataFrame({'text': texts, 'label': labels_with_noise})

                dup_idx = np.random.choice(n, size=15, replace=False)
                df.loc[dup_idx, 'text'] = df.loc[dup_idx, 'text']
                st.success(f'✅ 已生成 {len(df)} 条示例数据（含8%噪声）')

        if df is not None:
            text_candidates, label_candidates = get_text_and_label_candidates(df)
            default_text = text_candidates[0] if text_candidates else None
            default_label = label_candidates[0] if (label_candidates and len(label_candidates) > 0) else None

            text_col = st.selectbox('📝 文本列', options=text_candidates, index=0 if default_text else 0)
            label_col = st.selectbox('🏷️ 标签列', options=label_candidates, index=0 if default_label else 0)

            run_btn = st.button('🚀 运行完整诊断', type='primary')

            if run_btn:
                st.session_state['df'] = df
                st.session_state['text_col'] = text_col
                st.session_state['label_col'] = label_col
                st.session_state.pop('diagnostics', None)
                st.session_state.pop('split_result', None)

        if st.session_state['df'] is not None and st.session_state['diagnostics'] is None:
            with st.spinner('正在进行数据诊断（标签噪声检测、类别平衡分析、分布漂移检测、文本质量检查）...'):
                try:
                    diag = run_full_diagnostics(
                        st.session_state['df'],
                        st.session_state['text_col'],
                        st.session_state['label_col']
                    )
                    st.session_state['diagnostics'] = diag
                    st.success('✅ 诊断完成！')
                except Exception as e:
                    st.error(f'诊断失败: {str(e)}')
                    import traceback
                    st.code(traceback.format_exc())

    if st.session_state['df'] is None or st.session_state['diagnostics'] is None:
        st.info('👈 请在左侧栏上传数据、选择文本列和标签列，然后点击"运行完整诊断"')

        st.markdown('---')
        st.markdown('### 📖 使用说明')
        st.markdown('''
1. **上传数据**: 支持CSV/JSON/JSONL格式，或从HuggingFace直接加载数据集
2. **选择列**: 指定文本列和标签列
3. **运行诊断**: 自动执行以下检查：
   - 🏷️ 标签噪声检测（基于CleanLab交叉验证）
   - ⚖️ 类别不平衡分析（Gini不纯度）
   - ⚡ 分布漂移检测（JS散度）
   - 📝 文本质量检查（空文本、重复、乱码等）
4. **查看报告**: 四个Tab分别展示不同维度的诊断结果
5. **导出修正**: 支持批量修正标签并导出，一键生成HTML报告
        ''')
        return

    df = st.session_state['modified_df'] if st.session_state['modified_df'] is not None else st.session_state['df']
    diag = st.session_state['diagnostics']
    text_col = st.session_state['text_col']
    label_col = st.session_state['label_col']

    tab_overview, tab_noise, tab_balance, tab_drift = st.tabs([
        '📊 质量总览',
        '🏷️ 标签噪声',
        '⚖️ 类别平衡',
        '⚡ 分布漂移'
    ])

    with tab_overview:
        render_overview_tab(df, diag, text_col, label_col)

    with tab_noise:
        render_label_noise_tab(df, diag, label_col)

    with tab_balance:
        render_class_balance_tab(df, diag, text_col, label_col)

    with tab_drift:
        render_distribution_drift_tab(df, diag, text_col, label_col)


if __name__ == '__main__':
    main()
