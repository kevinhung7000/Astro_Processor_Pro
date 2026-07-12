#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_i18n_coverage.py
=======================
依 OPTIMIZATION_PLAN_v1.2.2.md「🧹 Process Note / 流程提醒」實作的靜態檢查腳本。

背景：v1.2.1 出過兩個問題，都是發版後複查才抓到，不是發版前：
  1. 4 個新元件完全沒接上語言切換機制 —— 也就是沒有被加進
     TRANSLATED_COMPONENTS_MAP 這個 (component變數, key) 對照表裡。
  2. 24 個滑桿的「標題」有翻譯、但「說明文字」沒有 —— 因為 UI_TRANSLATIONS
     字典裡對應的 key 存的是單純字串，不是 (label, info) 的 tuple。
  這兩個都是肉眼複查很容易漏掉的問題，因為元件建構時寫死的中文字串本來就
  看得懂、畫面也看起來完成了，只有切到 EN 之後才會發現某個元件沒變、或
  info 提示文字消失。

實際運作方式（重要，跟一般 gr.i18n 用法不同）：
  這支程式裡，gr.* 元件在建構當下用的多半是「寫死的中文字串」當 label/info
  （例如 label="本機資料夾路徑"），語言切換不是在建構時就讀字典，而是額外
  維護一份 TRANSLATED_COMPONENTS_MAP = [(component變數, "key"), ...] 清單，
  在語言切換的 callback 裡逐一用 gr.update(label=UI_TRANSLATIONS[key][lang])
  蓋掉畫面上的文字。所以「有沒有接上語言切換」不是看建構式裡有沒有引用
  UI_TRANSLATIONS，而是看這個元件變數有沒有出現在 TRANSLATED_COMPONENTS_MAP。

用法：
    python3 check_i18n_coverage.py Astro_Processor_Pro.py

檢查邏輯：
  1. 解析 TRANSLATED_COMPONENTS_MAP 字面量，取得所有已註冊的
     (元件變數名稱, UI_TRANSLATIONS key) pair。
  2. 掃描所有「變數 = gr.Xxx(...)」形式的元件建構式，記錄哪些元件在建構時
     帶了 label= 和/或 info=（純文字或非空字串），視為「這個元件的文字內容
     需要語言切換」的候選名單。
  3. 交叉比對：
     a) 候選元件的變數名稱如果沒出現在 TRANSLATED_COMPONENTS_MAP 裡 →
        「未接上語言切換」（對應 v1.2.1 的問題 1）。
     b) 元件建構時帶了 info=，且該元件確實有註冊到 TRANSLATED_COMPONENTS_MAP，
        但 UI_TRANSLATIONS[key]["zh"] 存的是單純字串、不是 (label, info) tuple
        → 「info 沒有被翻譯（bare string，不是 tuple）」（對應 v1.2.1 的問題 2）。

注意：這是啟發式（heuristic）靜態分析，不是完整的 gradio runtime 追蹤，
用於「發版前快速掃一輪」，不是取代人工複查。某些元件（例如語言切換鈕本身、
純裝飾用的 gr.Markdown 分隔線）刻意不進 TRANSLATED_COMPONENTS_MAP 是合理的，
腳本仍會列出來，需要人工確認是否為刻意排除，而非 bug。
"""
import ast
import sys


def find_assign_target_name(node: ast.AST) -> str | None:
    """從 AST 節點往上找『這個 Call 是不是被賦值給一個簡單變數名稱』。"""
    if isinstance(node, ast.Name):
        return node.id
    return None


def collect_translated_map(tree: ast.AST) -> set[tuple[str, str]]:
    """解析 TRANSLATED_COMPONENTS_MAP = [(var, "key"), ...] 取得 (變數名, key) 集合。"""
    pairs: set[tuple[str, str]] = set()

    class V(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "TRANSLATED_COMPONENTS_MAP":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                                comp_node, key_node = elt.elts
                                if isinstance(comp_node, ast.Name) and isinstance(key_node, ast.Constant) \
                                        and isinstance(key_node.value, str):
                                    pairs.add((comp_node.id, key_node.value))
            self.generic_visit(node)

    V().visit(tree)
    return pairs


def collect_gr_component_assignments(tree: ast.AST):
    """找出所有『var = gr.Xxx(...)』的元件建構式，回傳 list of dict。"""
    results = []

    class V(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            if len(node.targets) == 1 and isinstance(node.value, ast.Call):
                call = node.value
                func = call.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "gr":
                    var_name = find_assign_target_name(node.targets[0])
                    if var_name:
                        has_label = False
                        has_info = False
                        label_nonempty = False
                        for kw in call.keywords:
                            if kw.arg == "label":
                                has_label = True
                                if not (isinstance(kw.value, ast.Constant) and kw.value.value in ("", None)):
                                    label_nonempty = True
                            elif kw.arg == "info":
                                has_info = True
                        results.append({
                            "var": var_name,
                            "call": f"gr.{func.attr}",
                            "line": call.lineno,
                            "has_label": has_label,
                            "label_nonempty": label_nonempty,
                            "has_info": has_info,
                        })
            self.generic_visit(node)

    V().visit(tree)
    return results


def collect_ui_translations_tuple_info(tree: ast.AST) -> dict[str, bool]:
    """回傳 {key: zh_is_tuple}"""
    out: dict[str, bool] = {}

    class V(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "UI_TRANSLATIONS":
                    if isinstance(node.value, ast.Dict):
                        for k, v in zip(node.value.keys, node.value.values):
                            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                                continue
                            key = k.value
                            zh_is_tuple = False
                            if isinstance(v, ast.Dict):
                                for lk, lv in zip(v.keys, v.values):
                                    if isinstance(lk, ast.Constant) and lk.value == "zh":
                                        zh_is_tuple = isinstance(lv, (ast.Tuple, ast.List))
                            out[key] = zh_is_tuple
            self.generic_visit(node)

    V().visit(tree)
    return out


# 刻意不需要進語言切換表的元件（人工白名單，依實際情況調整）：
# lang_radio 本身是語言切換開關，theme_checkbox 是淺色模式切換，這兩個目前
# 是刻意保留固定文字（例如 "☀ Light Mode"），不是漏接。
KNOWN_INTENTIONAL_EXCLUSIONS = {"lang_radio", "theme_checkbox"}


def main():
    if len(sys.argv) != 2:
        print("用法: python3 check_i18n_coverage.py <path/to/Astro_Processor_Pro.py>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename=path)

    translated_map = collect_translated_map(tree)
    translated_vars = {var for var, _ in translated_map}
    var_to_key = dict(translated_map)
    components = collect_gr_component_assignments(tree)
    zh_tuple_info = collect_ui_translations_tuple_info(tree)

    candidates = [c for c in components if c["label_nonempty"] or c["has_info"]]

    not_registered = [
        c for c in candidates
        if c["var"] not in translated_vars and c["var"] not in KNOWN_INTENTIONAL_EXCLUSIONS
    ]
    info_not_tuple = []
    for c in candidates:
        if c["has_info"] and c["var"] in var_to_key:
            key = var_to_key[c["var"]]
            if key in zh_tuple_info and not zh_tuple_info[key]:
                info_not_tuple.append((c, key))

    print(f"掃描檔案：{path}")
    print(f"共找到 {len(components)} 個「var = gr.Xxx(...)」元件建構式，"
          f"其中 {len(candidates)} 個帶有非空 label 或 info（本次檢查候選）")
    print(f"TRANSLATED_COMPONENTS_MAP 已註冊 {len(translated_map)} 筆")
    print(f"UI_TRANSLATIONS 共登記 {len(zh_tuple_info)} 個 key\n")

    print("=" * 64)
    print(f"❌ 未接上語言切換機制（不在 TRANSLATED_COMPONENTS_MAP 裡）：{len(not_registered)} 個")
    print("=" * 64)
    for c in not_registered:
        print(f"  第 {c['line']:>5} 行  {c['var']:<30} {c['call']}")
    if not not_registered:
        print("  （無）")

    print()
    print("=" * 64)
    print(f"❌ info= 有值，但 UI_TRANSLATIONS 裡對應項目是單純字串、不是 tuple：{len(info_not_tuple)} 個")
    print("=" * 64)
    for c, key in info_not_tuple:
        print(f"  第 {c['line']:>5} 行  {c['var']:<30} key=\"{key}\"")
    if not info_not_tuple:
        print("  （無）")

    print()
    total = len(not_registered) + len(info_not_tuple)
    if total == 0:
        print("✅ 沒有發現這兩類已知問題模式（v1.2.1 那兩個 bug 的模式）。")
        sys.exit(0)
    else:
        print(f"⚠️ 共 {total} 項疑似問題，建議發版前逐一確認"
              f"（部分可能是刻意排除，例如純裝飾用元件；請對照白名單 KNOWN_INTENTIONAL_EXCLUSIONS）。")
        sys.exit(1)


if __name__ == "__main__":
    main()
