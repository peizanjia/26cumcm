"""Comparison UI using actual recorded actions and directional map frames."""
from question4.dynamic_joint.replay import render as dynamic_render


def render(data):
    html = dynamic_render(data)
    html = html.replace("run(side).summary.policy==='dynamic'",
                        "['dynamic','free_joint'].includes(run(side).summary.policy)")
    html = html.replace('第四问 · 动态未知域与跨源停点对照', '第四问 · 500秒目标完整验证')
    html = html.replace('动态未知域与跨源停点 · 完整运动历史', '自由补点与接收恢复 · 完整运动历史')
    html = html.replace("Object.assign(REASONS,{", "Object.assign(REASONS,{near_positive_anchor:'近端正反馈锚点探测',near_boundary_certified_clear:'可行清除区域边缘接近',")
    return html
