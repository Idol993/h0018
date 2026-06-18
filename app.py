import io
import json
import base64
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html
from typing import Optional, Dict, Any, List

from diagnostics import run_full_diagnostics, check_split_stratification, dataframe_fingerprint


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


def get_split_col_candidates(df: pd.DataFrame, text_col: str, label_col: str) -> List[str]:
    candidates = []
    for col in df.columns:
        if col == text_col or col == label_col:
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= min(20, len(df) * 0.5):
            candidates.append(col)
    return candidates


def highlight_rows(row):
    styles = [''] * len(row)
    if 'score_gap' in row.index and pd.notna(row['score_gap']):
        if row['score_gap'] > 0.5:
            styles = ['background-color: #ffcccc'] * len(row)
        elif row['score_gap'] > 0.3:
            styles = ['background-color: #ffeecc'] * len(row)
    return styles


def fig_to_svg(fig: go.Figure) -> str:
    """将Plotly图转为base64编码的SVG，用于HTML离线报告"""
    try:
        svg_bytes = fig.to_image(format='svg', width=800, height=500, scale=2)
        b64 = base64.b64encode(svg_bytes).decode()
        return f'<img src="data:image/svg+xml;base64,{b64}" style="max-width:100%; height:auto;" />'
    except Exception:
        try:
            html_div = to_html(fig, include_plotlyjs=False, full_html=False, div_id='chart')
            return html_div
        except Exception:
            return '<div style="padding:20px; background:#f5f5f5;">图表渲染失败</div>'


def plotly_js_inline() -> str:
    """返回内联的Plotly.js CDN引用，用于HTML报告"""
    return '''<script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>'''


def generate_pie_fig(class_counts: Dict[str, int]) -> go.Figure:
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = px.colors.qualitative.Set3[:len(classes)]
    fig = go.Figure(data=[go.Pie(
        labels=classes, values=counts, hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    fig.update_layout(title='各类别占比', height=450)
    return fig


def generate_text_quality_fig(tq: Dict[str, Any]) -> go.Figure:
    issue_types = ['空文本', '过短文本', '纯数字', '潜在乱码', '重复样本']
    issue_counts = [
        len(tq['empty_texts']),
        len(tq['too_short']),
        len(tq['pure_numeric']),
        len(tq['potential_gibberish']),
        len(tq['duplicates'])
    ]
    colors = ['#e74c3c' if c > 0 else '#95a5a6' for c in issue_counts]
    fig = go.Figure([go.Bar(x=issue_types, y=issue_counts, marker_color=colors, text=issue_counts, textposition='auto')])
    fig.update_layout(title='文本质量问题分布', height=450, yaxis_title='数量', yaxis=dict(showgrid=True))
    return fig


def generate_drift_fig(d: Dict[str, Any]) -> go.Figure:
    bins = d['len_hist_bins']
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=centers, y=d['len_hist_a'],
        name=d['split_label_a'], fill='tozeroy', mode='lines', line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=centers, y=d['len_hist_b'],
        name=d['split_label_b'], fill='tozeroy', mode='lines', line=dict(width=2)
    ))
    fig.update_layout(
        title=f'文本长度分布对比: {d["split_label_a"]} vs {d["split_label_b"]}',
        height=350,
        xaxis_title='文本长度（字符数）',
        yaxis_title='密度',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


def generate_class_bar_fig(class_counts: Dict[str, int], small_classes: Dict[str, int]) -> go.Figure:
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    small_flags = [c in small_classes for c in classes]
    colors = ['#f44336' if s else '#4f8bf9' for s in small_flags]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=classes, y=counts, marker_color=colors, text=counts, textposition='auto'))
    fig.update_layout(
        title='各类别样本数量（红色表示样本数<50）',
        height=450,
        xaxis_tickangle=-45,
        yaxis_title='样本数'
    )
    return fig


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
    lim = diagnostics_results.get('label_issues_meta', {})
    drift_meta = diagnostics_results.get('drift_meta', {})

    overall_color = '#4caf50' if qs['quality_score'] >= 0.7 else ('#ff9800' if qs['quality_score'] >= 0.5 else '#f44336')

    cleanlab_score_str = f'{qs["cleanlab_score"]:.1%}' if qs['cleanlab_score'] is not None else '未执行'
    noise_ratio_str = f'{qs["noise_ratio"]:.1%}' if qs['noise_ratio'] is not None else '未执行'

    drift_info_html = ''
    drift_charts_html = ''
    for idx, d in enumerate(drifts):
        len_color = 'color:red; font-weight:bold;' if d['text_length_js'] > 0.2 else 'color:green;'
        tf_color = 'color:red; font-weight:bold;' if d['tfidf_js'] > 0.2 else 'color:green;'
        drift_info_html += f'''
        <tr>
            <td>{d['split_label_a']} vs {d['split_label_b']}</td>
            <td style="{len_color}">{d['text_length_js']}</td>
            <td style="{tf_color}">{d['tfidf_js']}</td>
            <td>{d['size_a']} / {d['size_b']}</td>
        </tr>
        '''
        drift_fig = generate_drift_fig(d)
        drift_charts_html += f'''
        <div style="margin: 20px 0; padding: 15px; background:#f9fafc; border-radius:8px;">
            <h3 style="margin-top:0;">切分对比 {idx + 1}: {d["split_label_a"]} vs {d["split_label_b"]}</h3>
            {fig_to_svg(drift_fig)}
            <p style="color:#666; font-size:14px; margin-top:10px;">
                文本长度JS散度: <strong style="{len_color}">{d['text_length_js']}</strong> |
                TF-IDF词频JS散度: <strong style="{tf_color}">{d['tfidf_js']}</strong>
                {'  ⚠️ 存在显著漂移' if (d['text_length_js'] > 0.2 or d['tfidf_js'] > 0.2) else ''}
            </p>
        </div>
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
    all_suggestions = []
    all_suggestions.extend(cb['suggestions'])
    all_suggestions.extend(aug['strategies'])
    all_suggestions.extend(ss['warnings'])
    for d in drifts:
        if d['text_length_js'] > 0.2:
            all_suggestions.append(
                f"文本长度分布漂移过大 ({d['split_label_a']} vs {d['split_label_b']}): JS={d['text_length_js']}，建议检查数据来源一致性"
            )
        if d['tfidf_js'] > 0.2:
            all_suggestions.append(
                f"词频分布漂移过大 ({d['split_label_a']} vs {d['split_label_b']}): JS={d['tfidf_js']}，建议检查数据来源一致性"
            )
    if lim.get('skipped'):
        all_suggestions.append(f'ℹ️ 标签噪声检测: {lim.get("reason", "")}')
    elif lim.get('degraded'):
        all_suggestions.append(f'ℹ️ 标签噪声检测已降级: {lim.get("reason", "")}')

    for i, s in enumerate(all_suggestions[:25], 1):
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

    pie_fig = generate_pie_fig(cb['class_counts'])
    tq_fig = generate_text_quality_fig(tq)
    bar_fig = generate_class_bar_fig(cb['class_counts'], cb['small_classes'])

    drift_method = drift_meta.get('method', '默认前后切分')

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>标注数据质量诊断报告</title>
    {plotly_js_inline()}
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; margin: 40px auto; max-width: 1200px; color: #333; line-height: 1.6; }}
        h1 {{ color: #1a1a1a; border-bottom: 3px solid #4f8bf9; padding-bottom: 12px; margin-bottom: 20px; }}
        h2 {{ color: #2c3e50; margin-top: 35px; border-left: 4px solid #4f8bf9; padding-left: 12px; }}
        h3 {{ color: #34495e; }}
        .score-box {{ display: inline-block; padding: 24px 48px; background: {overall_color}; color: white; border-radius: 10px; font-size: 36px; font-weight: bold; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
        .score-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; margin: 25px 0; }}
        .score-card {{ background: #f5f7fa; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e4e9f0; }}
        .score-label {{ color: #666; font-size: 14px; margin-bottom: 8px; }}
        .score-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .score-sub {{ font-size: 12px; color: #999; margin-top: 4px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 10px 14px; text-align: left; }}
        th {{ background: #f0f4ff; color: #2c3e50; font-weight: 600; }}
        tr:hover {{ background: #f9fafc; }}
        .warning {{ color: #e67e22; font-weight: bold; }}
        .danger {{ color: #e74c3c; font-weight: bold; }}
        ul {{ line-height: 1.9; padding-left: 22px; }}
        .meta-box {{ background: #f8f9fa; padding: 15px 20px; border-radius: 6px; border-left: 4px solid #adb5bd; margin: 15px 0; }}
        .chart-container {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin: 15px 0; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin: 20px 0; }}
        .degrade-notice {{ background: #fff8e1; border-left: 4px solid #ffa726; padding: 12px 18px; margin: 10px 0; border-radius: 4px; }}
        @media (max-width: 800px) {{
            .score-grid {{ grid-template-columns: repeat(2, 1fr); }}
            .two-col {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>🔍 标注数据质量诊断报告</h1>
    <div class="meta-box">
        <p style="margin:5px 0;"><strong>📁 数据配置:</strong> 文本列 = "{text_col}"，标签列 = "{label_col}"</p>
        <p style="margin:5px 0;"><strong>📊 样本总数:</strong> {diagnostics_results['total_samples']} 条</p>
        <p style="margin:5px 0;"><strong>⚡ 漂移切分方式:</strong> {drift_method}</p>
    </div>

    <h2>📊 综合质量评分</h2>
    <div class="score-box">总体: {qs['quality_score']:.1%}</div>
    <div class="score-grid">
        <div class="score-card">
            <div class="score-label">CleanLab标注质量</div>
            <div class="score-value">{cleanlab_score_str}</div>
            <div class="score-sub">{'正常' if not lim.get('skipped') else lim.get('reason','')}</div>
        </div>
        <div class="score-card">
            <div class="score-label">噪声样本比例</div>
            <div class="score-value" class="{'danger' if qs['noise_ratio'] and qs['noise_ratio'] > 0.1 else ''}">{noise_ratio_str}</div>
            <div class="score-sub">可疑: {len(li)} 条</div>
        </div>
        <div class="score-card">
            <div class="score-label">类别平衡得分</div>
            <div class="score-value">{qs['balance_score']:.1%}</div>
            <div class="score-sub">Gini: {cb['gini_impurity']}</div>
        </div>
        <div class="score-card">
            <div class="score-label">类别数 / 小类别</div>
            <div class="score-value">{cb['num_classes']} / {len(cb['small_classes'])}</div>
            <div class="score-sub">最少: {cb['min_count']} 条</div>
        </div>
    </div>

    {f'<div class="degrade-notice">⚠️ {lim.get("reason", "")}</div>' if lim.get('degraded') or lim.get('skipped') else ''}

    <h2>🏷️ 类别分布</h2>
    <div class="two-col">
        <div class="chart-container">
            {fig_to_svg(pie_fig)}
        </div>
        <div class="chart-container">
            {fig_to_svg(bar_fig)}
        </div>
    </div>
    <table>
        <tr><th>类别</th><th>样本数</th><th>占比</th></tr>
        {class_rows}
    </table>
    <p style="color:#666; margin-top:8px;">
        <strong>Gini不纯度:</strong> <span class="{'danger' if cb['gini_impurity'] > 0.3 else ''}">{cb['gini_impurity']}</span>
        &nbsp;&nbsp;|&nbsp;&nbsp;
        <strong>最大/最小样本比:</strong> {cb['max_min_ratio']}x
    </p>

    <h2>📝 文本质量检查</h2>
    <div class="two-col">
        <div class="chart-container">
            {fig_to_svg(tq_fig)}
        </div>
        <div>
            <table>
                <tr><th>问题类型</th><th>数量</th></tr>
                {text_quality_summary}
            </table>
        </div>
    </div>

    <h2>⚡ 分布漂移检测 (JS散度)</h2>
    <p style="color:#666;">
        <strong>切分方式:</strong> {drift_method}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        阈值: 文本长度 JS > 0.2 或 词频 JS > 0.2 视为存在显著漂移
    </p>
    <table>
        <tr><th>对比组</th><th>文本长度JS散度</th><th>TF-IDF词频JS散度</th><th>样本数 (A / B)</th></tr>
        {drift_info_html}
    </table>
    {drift_charts_html}

    <h2>💡 修复与优化建议</h2>
    <ul>{suggestions_html}</ul>

    <p style="margin-top: 50px; color: #999; text-align: center; font-size: 13px;">
        标注数据质量诊断报告 — 由 Streamlit + CleanLab 生成
    </p>
</body>
</html>'''
    return html


def get_download_link(html_content: str, filename: str = 'diagnostics_report.html') -> str:
    b64 = base64.b64encode(html_content.encode('utf-8')).decode()
    return f'<a href="data:text/html;charset=utf-8;base64,{b64}" download="{filename}" style="display:inline-block;padding:10px 20px;background:#4f8bf9;color:white;text-decoration:none;border-radius:6px;font-weight:500;">📥 下载HTML诊断报告</a>'


def render_overview_tab(df: pd.DataFrame, diag: Dict[str, Any], text_col: str, label_col: str):
    qs = diag['quality_scores']
    lim = diag.get('label_issues_meta', {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric('综合质量评分', f'{qs["quality_score"]:.1%}')
    with col2:
        if qs['cleanlab_score'] is not None:
            st.metric('标注质量(CleanLab)', f'{qs["cleanlab_score"]:.1%}')
        else:
            st.metric('标注质量', '未执行')
            st.caption(lim.get('reason', ''))
    with col3:
        if qs['noise_ratio'] is not None:
            st.metric('噪声样本比例', f'{qs["noise_ratio"]:.1%}', delta=f'{len(diag["label_issues"])}条')
        else:
            st.metric('噪声样本', '未检测')
    with col4:
        st.metric('类别平衡得分', f'{qs["balance_score"]:.1%}')

    if lim.get('degraded') or lim.get('skipped'):
        st.info(f'ℹ️ 标签噪声检测状态: {lim.get("reason", "")}')

    st.markdown('---')

    cb = diag['class_balance']
    tq = diag['text_quality']
    drifts = diag['distribution_drift']
    drift_meta = diag.get('drift_meta', {})

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader('📈 类别分布概览')
        pie_fig = generate_pie_fig(cb['class_counts'])
        st.plotly_chart(pie_fig, use_container_width=True)

    with col_b:
        st.subheader('⚠️ 文本质量问题统计')
        tq_fig = generate_text_quality_fig(tq)
        st.plotly_chart(tq_fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📊 分布漂移检测')
    st.caption(f'切分方式: {drift_meta.get("method", "默认")}')

    if not drifts:
        st.info('未检测到有效的漂移对比组（可能样本太少或切分列取值不足）')
    else:
        for d in drifts:
            with st.expander(f'**{d["split_label_a"]} vs {d["split_label_b"]}**  (文本长度 JS={d["text_length_js"]:.4f} / 词频 JS={d["tfidf_js"]:.4f})', expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    color_js_len = '🔴 显著漂移' if d['text_length_js'] > 0.2 else '🟢 正常'
                    st.metric(f'{color_js_len} 文本长度JS散度', f"{d['text_length_js']:.4f}")
                with c2:
                    color_js_tf = '🔴 显著漂移' if d['tfidf_js'] > 0.2 else '🟢 正常'
                    st.metric(f'{color_js_tf} TF-IDF词频JS散度', f"{d['tfidf_js']:.4f}")
                with c3:
                    st.metric('样本数 (A/B)', f"{d['size_a']} / {d['size_b']}")
                    st.caption(f'平均长度: {d["avg_len_a"]:.0f} / {d["avg_len_b"]:.0f}')

                drift_fig = generate_drift_fig(d)
                st.plotly_chart(drift_fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📄 一键导出诊断报告')
    html_report = generate_html_report(df, diag, text_col, label_col)
    st.markdown(get_download_link(html_report), unsafe_allow_html=True)
    st.caption('HTML报告包含交互式图表（需联网加载Plotly.js），并内置SVG备用图可离线查看')


def render_label_noise_tab(df: pd.DataFrame, diag: Dict[str, Any], label_col: str, text_col: str):
    label_issues = diag['label_issues']
    lim = diag.get('label_issues_meta', {})

    if lim.get('skipped'):
        st.warning(f'⚠️ 标签噪声检测已跳过: {lim.get("reason", "")}')
        st.info('类别样本太少时无法进行交叉验证噪声检测，请补充数据后重试')
        return

    if lim.get('degraded'):
        st.info(f'ℹ️ 标签噪声检测已降级运行: {lim.get("reason", "")}')

    if len(label_issues) == 0:
        st.success('🎉 未发现明显的标签噪声样本，标注质量看起来不错！')
        return

    st.markdown(f'### 🔍 发现 {len(label_issues)} 条可疑标注样本')
    st.info('💡 **score_gap** 表示模型建议标签与原标签的置信度差距，数值越大越可能是标注错误')

    sort_by = st.selectbox(
        '排序方式',
        ['可疑程度（降序）', '可疑程度（升序）', '原标签', '建议标签'],
        index=0,
        key='noise_sort'
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
    st.subheader('✏️ 批量修正标签并重新诊断')

    all_labels = sorted(df[label_col].astype(str).unique().tolist())
    if len(label_issues) > 0:
        suggested_labels = sorted(label_issues['suggested_label'].unique().tolist())
        for sl in suggested_labels:
            if sl not in all_labels:
                all_labels.append(sl)

    selected_indices = st.multiselect(
        '选择要修正的样本（按数据集中的原始索引）',
        options=label_issues['index'].tolist(),
        format_func=lambda x: f"索引#{x}: {str(label_issues[label_issues['index'] == x]['text'].values[0])[:60]}...",
        key='noise_selected'
    )

    if selected_indices:
        col1, col2 = st.columns(2)
        with col1:
            first_idx = selected_indices[0]
            first_suggested = label_issues[label_issues['index'] == first_idx]['suggested_label'].values[0]
            suggested_default = all_labels.index(first_suggested) if first_suggested in all_labels else 0
            new_label = st.selectbox('新标签', options=all_labels, index=suggested_default, key='new_label_select')
        with col2:
            st.write('')
            apply_btn = st.button('✅ 应用修正并重新诊断', type='primary', key='apply_fix_btn')

        if apply_btn:
            modified_df = df.copy()
            modified_df.loc[selected_indices, label_col] = new_label
            st.session_state['df'] = modified_df
            st.session_state['modified_df'] = modified_df
            st.session_state.pop('diagnostics', None)
            st.session_state.pop('split_result', None)
            st.session_state.pop('drift_split_col', None)
            st.success(f'✅ 已将 {len(selected_indices)} 条样本的标签修正为 "{new_label}"，正在重新诊断...')

            with st.spinner('🔄 基于修正后的数据重新运行完整诊断...'):
                try:
                    split_col = st.session_state.get('drift_split_col', None)
                    drift_mode = st.session_state.get('drift_mode', 'auto')
                    new_diag = run_full_diagnostics(
                        modified_df, text_col, label_col,
                        split_col=split_col, drift_mode=drift_mode
                    )
                    st.session_state['diagnostics'] = new_diag
                    st.success('🎉 重新诊断完成！所有数据已更新')
                    st.balloons()
                except Exception as e:
                    st.error(f'重新诊断失败: {str(e)}')

            st.rerun()

    st.markdown('---')
    col_export1, col_export2 = st.columns(2)
    with col_export1:
        st.download_button(
            label='📥 导出可疑样本清单 (CSV)',
            data=label_issues.to_csv(index=False).encode('utf-8-sig'),
            file_name='suspicious_labels.csv',
            mime='text/csv',
            key='export_suspicious'
        )
    with col_export2:
        st.download_button(
            label='📥 导出当前完整数据 (CSV)',
            data=df.to_csv(index=False).encode('utf-8-sig'),
            file_name='current_dataset.csv',
            mime='text/csv',
            key='export_current'
        )


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

    if len(cb.get('tiny_classes', {})) > 0:
        tiny = ', '.join(cb['tiny_classes'].keys())
        st.error(f'⚠️ 以下类别样本极少 (<5条): {tiny}，标签噪声检测可能已降级或跳过')

    st.markdown('---')
    st.subheader('📊 类别分布可视化')

    classes = list(cb['class_counts'].keys())
    counts = list(cb['class_counts'].values())
    props = [cb['class_proportions'][c] for c in classes]
    small_flags = [c in cb['small_classes'] for c in classes]

    bar_fig = generate_class_bar_fig(cb['class_counts'], cb['small_classes'])
    st.plotly_chart(bar_fig, use_container_width=True)

    pie_fig = generate_pie_fig(cb['class_counts'])
    pie_fig.update_layout(title='各类别占比（点击图例可下钻查看）', height=550)
    st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown('---')
    st.subheader('📋 类别分布明细')

    detail_df = pd.DataFrame({
        '类别': classes,
        '样本数': counts,
        '占比': [f'{p:.2%}' for p in props],
        '是否小类别(<50)': ['🔴 是' if s else '🟢 否' for s in small_flags],
        '是否极少(<5)': ['🔴 是' if c in cb.get('tiny_classes', {}) else '🟢 否' for c in classes]
    })
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.subheader('💡 类别平衡建议')

    if cb['suggestions']:
        for i, s in enumerate(cb['suggestions'], 1):
            if '⚠️' in s:
                st.error(f'**警告 {i}:** {s}')
            else:
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
    st.subheader('📊 切分策略检查（训练/验证/测试）')

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider('训练集比例', 0.5, 0.9, 0.7, 0.05, key='split_train')
    with col2:
        val_ratio = st.slider('验证集比例', 0.05, 0.3, 0.15, 0.05, key='split_val')
    with col3:
        test_ratio = st.slider('测试集比例', 0.05, 0.3, 0.15, 0.05, key='split_test')

    split_key = f'{train_ratio}_{val_ratio}_{test_ratio}'
    if ('split_result' not in st.session_state
        or st.session_state.get('split_key') != split_key
        or st.session_state.get('split_data_hash') != hash(str(df[label_col].tolist()))):
        split_result = check_split_stratification(
            df[label_col].astype(str).tolist(),
            train_size=train_ratio,
            val_size=val_ratio,
            test_size=test_ratio
        )
        st.session_state['split_result'] = split_result
        st.session_state['split_key'] = split_key
        st.session_state['split_data_hash'] = hash(str(df[label_col].tolist()))
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
        height=450,
        xaxis_tickangle=-45,
        yaxis_title='占比 (%)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')
    st.subheader('⚡ 分布漂移检测（自定义切分列）')

    split_candidates = get_split_col_candidates(df, text_col, label_col)
    drift_meta = diag.get('drift_meta', {})

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        default_idx = 0
        if 'drift_split_col' in st.session_state and st.session_state['drift_split_col'] in split_candidates:
            default_idx = split_candidates.index(st.session_state['drift_split_col'])
        split_col_choice = st.selectbox(
            '选择漂移切分列（来源/时间/split等）',
            options=['(默认: 前后半切分)'] + split_candidates,
            index=default_idx if split_candidates else 0,
            help='选择一个列来划分两组数据，比较它们的文本长度和词频分布差异',
            key='drift_split_col_select'
        )
    with col_s2:
        drift_mode = st.selectbox(
            '对比模式',
            options=['auto（自动选择）', 'pairwise（两两对比）', 'first_vs_rest（第一个vs其他）'],
            index=0,
            key='drift_mode_select'
        )

    mode_map = {
        'auto（自动选择）': 'auto',
        'pairwise（两两对比）': 'pairwise',
        'first_vs_rest（第一个vs其他）': 'first_vs_rest'
    }
    selected_mode = mode_map.get(drift_mode, 'auto')
    actual_split_col = None if split_col_choice == '(默认: 前后半切分)' else split_col_choice

    st.session_state['drift_split_col'] = actual_split_col
    st.session_state['drift_mode'] = selected_mode

    prev_split_col = st.session_state.get('last_drift_split_col', '__none__')
    prev_mode = st.session_state.get('last_drift_mode', '__none__')

    if actual_split_col != prev_split_col or selected_mode != prev_mode:
        st.session_state['last_drift_split_col'] = actual_split_col
        st.session_state['last_drift_mode'] = selected_mode
        st.session_state.pop('diagnostics', None)
        st.rerun()

    drifts = diag['distribution_drift']

    st.caption(f'当前切分方式: {drift_meta.get("method", "未设置")}')

    if not drifts:
        st.info('未检测到有效的漂移对比组（可能样本太少或切分列取值不足）')
    else:
        for i, d in enumerate(drifts):
            st.markdown(f'#### 切分 {i + 1}: {d["split_label_a"]} vs {d["split_label_b"]}')

            c1, c2, c3 = st.columns(3)
            with c1:
                js_len = d['text_length_js']
                status_len = '🔴 显著漂移' if js_len > 0.2 else '🟢 正常'
                st.metric(f'文本长度 JS散度: {status_len}', f'{js_len:.4f}')
            with c2:
                js_tf = d['tfidf_js']
                status_tf = '🔴 显著漂移' if js_tf > 0.2 else '🟢 正常'
                st.metric(f'TF-IDF词频 JS散度: {status_tf}', f'{js_tf:.4f}')
            with c3:
                st.metric('样本数 (A / B)', f"{d['size_a']} / {d['size_b']}")
                st.caption(f'平均长度: {d["avg_len_a"]:.0f} / {d["avg_len_b"]:.0f}')

            if js_len > 0.2 or js_tf > 0.2:
                st.warning(f'⚠️ 该切分存在显著分布漂移，可能影响模型泛化能力')

            drift_fig = generate_drift_fig(d)
            st.plotly_chart(drift_fig, use_container_width=True)


def get_dataset_fingerprint(df: pd.DataFrame, text_col: str, label_col: str,
                            split_col: Optional[str], drift_mode: str) -> str:
    """生成数据集的指纹，用于判断是否需要重新诊断"""
    base_fp = dataframe_fingerprint(df, text_col, label_col)
    extra = f'_{split_col}_{drift_mode}'
    return hashlib.md5((base_fp + extra).encode('utf-8')).hexdigest()


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
    if 'data_fingerprint' not in st.session_state:
        st.session_state['data_fingerprint'] = None
    if 'drift_split_col' not in st.session_state:
        st.session_state['drift_split_col'] = None
    if 'drift_mode' not in st.session_state:
        st.session_state['drift_mode'] = 'auto'

    with st.sidebar:
        st.header('📂 数据输入')

        input_mode = st.radio(
            '选择数据输入方式',
            ['上传本地文件', 'HuggingFace数据集', '使用示例数据'],
            index=0,
            key='input_mode'
        )

        df = None

        if input_mode == '上传本地文件':
            uploaded_file = st.file_uploader(
                '拖入或选择标注文件',
                type=['csv', 'json', 'jsonl'],
                help='支持CSV、JSON数组、JSON Lines格式，选好文本列和标签列后自动开始诊断',
                key='file_uploader'
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
            hf_name = st.text_input('HuggingFace数据集名称', placeholder='例如: imdb, ag_news', key='hf_name')
            hf_split = st.text_input('数据集split', value='train', key='hf_split')
            if st.button('📥 加载数据集', key='load_hf'):
                if hf_name:
                    with st.spinner('正在从HuggingFace加载...'):
                        df = load_huggingface(hf_name, hf_split)
                        if df is not None:
                            st.success(f'✅ 成功加载 {len(df)} 条数据')

        elif input_mode == '使用示例数据':
            st.info('使用内置的示例数据（模拟分类标注，含噪声和各类质量问题）')
            if st.button('🔄 生成示例数据', type='primary', key='gen_sample'):
                np.random.seed(42)
                n = 500
                labels = np.random.choice(['正面', '负面', '中性'], size=n, p=[0.5, 0.3, 0.2])
                sources = np.random.choice(['微博', '知乎', '小红书'], size=n, p=[0.4, 0.35, 0.25])
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

                df = pd.DataFrame({'text': texts, 'label': labels_with_noise, 'source': sources})

                dup_idx = np.random.choice(n, size=15, replace=False)
                df.loc[dup_idx, 'text'] = df.loc[dup_idx, 'text']
                st.success(f'✅ 已生成 {len(df)} 条示例数据（含约8%噪声，含source来源列）')

        if df is not None:
            text_candidates, label_candidates = get_text_and_label_candidates(df)
            default_text = text_candidates[0] if text_candidates else None
            default_label = label_candidates[0] if (label_candidates and len(label_candidates) > 0) else None

            text_col = st.selectbox(
                '📝 文本列',
                options=text_candidates,
                index=0 if default_text else 0,
                key='text_col_sel'
            )
            label_col = st.selectbox(
                '🏷️ 标签列',
                options=label_candidates,
                index=0 if default_label else 0,
                key='label_col_sel'
            )

            current_fp = get_dataset_fingerprint(
                df, text_col, label_col,
                st.session_state.get('drift_split_col'),
                st.session_state.get('drift_mode', 'auto')
            )

            is_new_data = (
                st.session_state['df'] is None
                or st.session_state['data_fingerprint'] != current_fp
            )

            if is_new_data:
                st.session_state['df'] = df
                st.session_state['text_col'] = text_col
                st.session_state['label_col'] = label_col
                st.session_state['data_fingerprint'] = current_fp
                st.session_state.pop('diagnostics', None)
                st.session_state.pop('split_result', None)
                st.session_state['modified_df'] = None

            st.info('📊 选择文本列和标签列后将自动开始诊断')

    df = st.session_state['df']
    text_col = st.session_state['text_col']
    label_col = st.session_state['label_col']

    if df is None or text_col is None or label_col is None:
        st.info('👈 请在左侧栏上传数据、选择文本列和标签列')
        st.markdown('---')
        st.markdown('### 📖 使用说明')
        st.markdown('''
1. **上传数据**: 支持CSV/JSON/JSONL格式，或从HuggingFace直接加载数据集
2. **选择列**: 指定文本列和标签列，选好后自动开始诊断
3. **自动缓存**: 重复上传同一份数据不会重复计算，换列或换文件才会重新诊断
4. **四个诊断维度**:
   - 📊 **质量总览**: 综合评分、类别分布、文本质量、漂移检测、HTML报告导出
   - 🏷️ **标签噪声**: CleanLab检测可疑标注，支持批量修正并自动重跑
   - ⚖️ **类别平衡**: Gini不纯度、小类别识别、文本增强策略建议
   - ⚡ **分布漂移**: 训练/验证/测试分层抽样检查，自定义切分列漂移检测
        ''')
        return

    if st.session_state.get('diagnostics') is None:
        with st.spinner('🔬 正在运行完整诊断（标签噪声检测、类别平衡分析、分布漂移检测、文本质量检查）...'):
            try:
                split_col = st.session_state.get('drift_split_col')
                drift_mode = st.session_state.get('drift_mode', 'auto')
                diag = run_full_diagnostics(
                    df, text_col, label_col,
                    split_col=split_col,
                    drift_mode=drift_mode
                )
                st.session_state['diagnostics'] = diag
                st.toast('✅ 诊断完成！', icon='🎉')
            except Exception as e:
                st.error(f'诊断失败: {str(e)}')
                import traceback
                st.code(traceback.format_exc())
                return

    diag = st.session_state['diagnostics']

    tab_overview, tab_noise, tab_balance, tab_drift = st.tabs([
        '📊 质量总览',
        '🏷️ 标签噪声',
        '⚖️ 类别平衡',
        '⚡ 分布漂移'
    ])

    with tab_overview:
        render_overview_tab(df, diag, text_col, label_col)

    with tab_noise:
        render_label_noise_tab(df, diag, label_col, text_col)

    with tab_balance:
        render_class_balance_tab(df, diag, text_col, label_col)

    with tab_drift:
        render_distribution_drift_tab(df, diag, text_col, label_col)


if __name__ == '__main__':
    main()
