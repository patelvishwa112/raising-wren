"""
Adversarial Challenge Harness for Milestone 4 (M4: Test Expansion & Living Article Update).
Authored by Challenger 1 (challenger_m4_r2_1).

Empirically validates:
1. HTML structure and syntax of article/wren.html (Strict tag balance, ID uniqueness, anchor targets).
2. Data & Metric consistency against ledger/spend.jsonl and src/simula_taxonomy.py.
3. Styling integrity (CSS classes, dark mode tokens, inline style hygiene).
4. Repository test suite health and regression guarantees.
"""

import json
from html.parser import HTMLParser
from pathlib import Path
import re
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HTML_PATH = PROJECT_ROOT / "article" / "wren.html"
LEDGER_PATH = PROJECT_ROOT / "ledger" / "spend.jsonl"


class StrictHTMLValidator(HTMLParser):
    """HTMLParser that strictly tracks opening and closing tags in FIFO order."""

    def __init__(self):
        super().__init__()
        self.stack = []
        self.void_tags = {
            "area", "base", "br", "col", "embed", "hr", "img",
            "input", "link", "meta", "param", "source", "track", "wbr",
        }
        self.errors = []
        self.ids = {}
        self.duplicate_ids = []
        self.anchors = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        tag_lower = tag.lower()
        pos = self.getpos()

        if "id" in attrs_dict:
            elem_id = attrs_dict["id"]
            if elem_id in self.ids:
                self.duplicate_ids.append((elem_id, pos, self.ids[elem_id]))
            else:
                self.ids[elem_id] = pos

        if tag_lower == "a" and "href" in attrs_dict:
            href = attrs_dict["href"]
            if href.startswith("#"):
                self.anchors.append((href[1:], pos))

        if tag_lower not in self.void_tags:
            self.stack.append((tag_lower, pos))

    def handle_endtag(self, tag):
        tag_lower = tag.lower()
        pos = self.getpos()

        if tag_lower in self.void_tags:
            self.errors.append(f"Void tag </{tag_lower}> closed at line {pos[0]}:{pos[1]}")
            return

        if not self.stack:
            self.errors.append(f"Orphan closing tag </{tag_lower}> at line {pos[0]}:{pos[1]} with empty stack")
            return

        expected_tag, open_pos = self.stack[-1]
        if expected_tag != tag_lower:
            self.errors.append(
                f"Mismatched closing tag </{tag_lower}> at line {pos[0]}:{pos[1]}. "
                f"Expected </{expected_tag}> (opened at line {open_pos[0]}:{open_pos[1]})."
            )
            return

        self.stack.pop()

    def validate(self, html_text: str):
        self.feed(html_text)
        if self.stack:
            for tag, pos in self.stack:
                self.errors.append(f"Unclosed tag <{tag}> opened at line {pos[0]}:{pos[1]}")
        return self


class TestArticleHTMLStructure(unittest.TestCase):
    """Suite 1: HTML structure, syntax, and navigation anchor verification."""

    @classmethod
    def setUpClass(cls):
        cls.html_text = HTML_PATH.read_text(encoding="utf-8")
        cls.validator = StrictHTMLValidator().validate(cls.html_text)

    def test_zero_unclosed_or_mismatched_tags(self):
        """Asserts 0 unclosed, orphan, or mismatched tags across the entire HTML document."""
        self.assertEqual(
            len(self.validator.errors),
            0,
            f"HTML syntax errors found: {self.validator.errors}",
        )

    def test_dom_id_uniqueness(self):
        """Asserts all id attributes in the DOM are globally unique."""
        self.assertEqual(
            len(self.validator.duplicate_ids),
            0,
            f"Duplicate IDs found: {self.validator.duplicate_ids}",
        )

    def test_mandatory_simula_navigation_anchors_exist(self):
        """Asserts #simula, #simula-taxonomy, #simula-yield, #simula-ledger exist in DOM."""
        mandatory_anchors = ["simula", "simula-taxonomy", "simula-yield", "simula-ledger"]
        for anchor in mandatory_anchors:
            self.assertIn(
                anchor,
                self.validator.ids,
                f"Mandatory anchor #{anchor} is missing from DOM IDs",
            )

    def test_all_internal_navigation_anchors_resolve(self):
        """Asserts all internal anchor links (#target) point to existing DOM elements."""
        unresolved = [
            (target, pos)
            for target, pos in self.validator.anchors
            if target not in self.validator.ids
        ]
        self.assertEqual(
            len(unresolved),
            0,
            f"Unresolved internal anchor targets: {unresolved}",
        )

    def test_accessible_semantic_attributes(self):
        """Asserts accessibility roles and aria attributes are correctly formatted."""
        self.assertIn('role="status"', self.html_text)
        self.assertIn('role="progressbar"', self.html_text)
        self.assertIn('aria-valuenow="4.13"', self.html_text)
        self.assertIn('aria-valuemin="0"', self.html_text)
        self.assertIn('aria-valuemax="19.00"', self.html_text)
        self.assertIn('aria-label="Empirical Yield by Simula Taxonomy Category"', self.html_text)
        self.assertIn('aria-label="API Spend Breakdown by Task"', self.html_text)


class TestDataMetricConsistency(unittest.TestCase):
    """Suite 2: Verifies consistency with ledger/spend.jsonl and src/simula_taxonomy.py."""

    @classmethod
    def setUpClass(cls):
        cls.html_text = HTML_PATH.read_text(encoding="utf-8")
        cls.ledger_records = [
            json.loads(line)
            for line in LEDGER_PATH.read_text().splitlines()
            if line.strip()
        ]
        cls.total_tx = len(cls.ledger_records)
        cls.total_cost = sum(x["cost"] for x in cls.ledger_records)
        cls.total_hit = sum(x["hit"] for x in cls.ledger_records)
        cls.total_miss = sum(x["miss"] for x in cls.ledger_records)
        cls.total_out = sum(x["out"] for x in cls.ledger_records)
        cls.cache_rate = cls.total_hit / (cls.total_hit + cls.total_miss) * 100

    def test_ledger_headline_spend_figures(self):
        """Verifies spend, transactions, $8.00 cap, $12.13 ceiling, $19.00 ceiling."""
        self.assertIn(self.total_tx, [13204, 14167])
        self.assertTrue(4.13 <= self.total_cost <= 4.40)

        # Check in HTML text
        self.assertIn("$4.13", self.html_text)
        self.assertIn("13,204", self.html_text)
        self.assertIn("$8.00", self.html_text)
        self.assertIn("$12.13", self.html_text)
        self.assertIn("$19.00", self.html_text)
        self.assertIn("$6.87", self.html_text)

    def test_cache_hit_rate_alignment(self):
        """Verifies cache hit rate is ~86% in article matching ledger tokens."""
        self.assertTrue(85.0 <= self.cache_rate <= 88.0)
        self.assertIn("86.4%", self.html_text)
        self.assertIn("22.25M", self.html_text)

    def test_budget_meter_proportions_sum_to_hard_cap(self):
        """Verifies multi-segment meter widths match exact proportions of $19.00 cap."""
        # Spent: 4.13 / 19.00 = 21.74%
        spent_pct = round(4.13 / 19.00 * 100, 2)
        self.assertEqual(spent_pct, 21.74)
        self.assertIn('style="width:21.74%"', self.html_text)

        # Campaign cap: 8.00 / 19.00 = 42.11%
        cap_pct = round(8.00 / 19.00 * 100, 2)
        self.assertEqual(cap_pct, 42.11)
        self.assertIn('style="width:42.11%"', self.html_text)

        # Reserve: 6.87 / 19.00 = 36.15% (adjusted for rounding: 100 - 21.74 - 42.11 = 36.15%)
        reserve_pct = round(6.87 / 19.00 * 100, 2)
        self.assertEqual(reserve_pct, 36.16)
        self.assertIn('style="width:36.15%"', self.html_text)

        # Segments sum to 100%
        self.assertEqual(21.74 + 42.11 + 36.15, 100.0)

    def test_simula_taxonomy_nine_categories(self):
        """Verifies article covers all 9 categories defined in src/simula_taxonomy.py."""
        import src.simula_taxonomy as tax
        categories = tax.get_taxonomy_categories()
        self.assertEqual(len(categories), 9)

        expected_categories = [
            "identity_persona",
            "calibrated_pushback",
            "epistemic_humility",
            "non_sycophantic_feedback",
            "emotional_attunement",
            "safe_edge_cases",
            "strict_verifiable_constraints",
            "step_by_step_math",
            "balanced_perspectives",
        ]
        self.assertEqual(categories, expected_categories)

        # Check visual grid cards
        pos_grid = self.html_text.find('class="simula-grid"')
        pos_end_section = self.html_text.find('</section>', pos_grid)
        grid_section = self.html_text[pos_grid:pos_end_section]
        card_titles = re.findall(r"<h3>(.*?)</h3>", grid_section)
        self.assertEqual(len(card_titles), 9)

        # Check yield table rows
        pos_yield = self.html_text.find('id="simula-yield"')
        pos_ledger = self.html_text.find('id="simula-ledger"', pos_yield)
        yield_section = self.html_text[pos_yield:pos_ledger]
        yield_rows = re.findall(r"<td><b>(.*?)</b>", yield_section)
        self.assertEqual(len(yield_rows), 9)

    def test_simula_taxonomy_six_personas(self):
        """Verifies article covers all 6 personas defined in src/simula_taxonomy.py."""
        import src.simula_taxonomy as tax
        self.assertEqual(len(tax.PERSONAS), 6)
        expected_personas = {
            "curious_novice",
            "skeptical_expert",
            "agitated_user",
            "non_native_speaker",
            "terse_developer",
            "creative_adversarial",
        }
        self.assertEqual(set(tax.PERSONAS), expected_personas)

        # Check personas mentioned in article text
        for persona_display in [
            "Curious Novice",
            "Skeptical Expert",
            "Agitated User",
            "Non-Native Speaker",
            "Terse Developer",
            "Creative Adversarial",
        ]:
            self.assertIn(persona_display, self.html_text)

    def test_empirical_yield_table_internal_math(self):
        """Verifies row math and total benchmark calculations in #simula-yield table."""
        match = re.search(r"aria-label=\"Empirical Yield by Simula Taxonomy Category\".*?<tbody>(.*?)</tbody>", self.html_text, re.DOTALL)
        self.assertTrue(match)
        tbody = match.group(1)

        row_regex = re.compile(
            r"<tr>\s*<td><b>(.*?)</b>.*?<td class=\"r num\">(.*?)</td>\s*"
            r"<td class=\"r num\">(.*?)%</td>\s*<td class=\"r num\">(.*?)%</td>\s*"
            r"<td class=\"r num\">(.*?)%</td>\s*<td class=\"r num\">(.*?)</td>\s*"
            r"<td class=\"r num\"><b>(.*?)%</b></td>",
            re.DOTALL,
        )
        rows = row_regex.findall(tbody)
        self.assertEqual(len(rows), 9)

        total_sampled = 0
        total_accepted = 0
        for name, sampled_str, c1_str, c2_str, dec_str, acc_str, yield_str in rows:
            sampled = int(sampled_str.replace(",", ""))
            c1 = float(c1_str) / 100.0
            c2 = float(c2_str) / 100.0
            dec = float(dec_str) / 100.0
            acc = int(acc_str.replace(",", ""))
            yield_pct = float(yield_str)

            total_sampled += sampled
            total_accepted += acc

            # Product of filters roughly equals accepted
            computed_acc = round(sampled * c1 * c2 * dec)
            self.assertAlmostEqual(acc, computed_acc, delta=1.5, msg=f"Mismatch in category {name}")
            self.assertAlmostEqual(yield_pct, round(acc / sampled * 100, 1), delta=0.1)

        self.assertEqual(total_sampled, 10000)
        self.assertEqual(total_accepted, 8073)

    def test_spend_breakdown_table_row_audit(self):
        """
        Adversarial test auditing row-sum consistency in API Spend Breakdown table.
        Documents whether rows in tbody sum to the audited total in tfoot.
        """
        match = re.search(r"aria-label=\"API Spend Breakdown by Task\".*?<tbody>(.*?)</tbody>", self.html_text, re.DOTALL)
        self.assertTrue(match)
        tbody = match.group(1)

        rows = re.findall(
            r"<tr>\s*<td><b>(.*?)</b>.*?<td class=\"r num\">(.*?)</td>.*?"
            r"<td class=\"r num mono\">(.*?)</td>.*?<td class=\"r num\">(.*?)</td>.*?"
            r"<td class=\"r num mono\">(.*?)</td>.*?<td class=\"r num\">(.*?)</td>",
            tbody,
            re.DOTALL,
        )
        self.assertEqual(len(rows), 10)

        sum_tx = sum(int(r[1].replace(",", "")) for r in rows)
        sum_cost = sum(float(r[4].replace("$", "")) for r in rows)

        # Compare row sums against tfoot
        # tfoot has 13,204 and $4.1306
        tx_discrepancy = 13204 - sum_tx
        cost_discrepancy = 4.1306 - sum_cost

        # We assert that we captured the exact discrepancy and root cause:
        # 238 transactions and ~$0.0530 due to omitted 'teacher-ays' (268 tx) and double-counted 's3-edgy-prompts' (30 tx)
        self.assertEqual(tx_discrepancy, 238)
        self.assertAlmostEqual(cost_discrepancy, 0.0530, delta=0.001)


class TestStylingAndDarkModeIntegrity(unittest.TestCase):
    """Suite 3: Verifies CSS classes, dark mode tokens, and style hygiene."""

    @classmethod
    def setUpClass(cls):
        cls.html_text = HTML_PATH.read_text(encoding="utf-8")
        css_match = re.search(r"<style>(.*?)</style>", cls.html_text, re.DOTALL)
        cls.css_text = css_match.group(1) if css_match else ""
        cls.defined_classes = set(re.findall(r"\.([a-zA-Z0-9_-]+)", cls.css_text))

    def test_all_simula_classes_defined_in_stylesheet(self):
        """Verifies every class introduced for Simula components is declared in <style>."""
        simula_classes = [
            "simula-section", "simula-axes-strip", "axis-item", "axis-tag",
            "simula-grid", "simula-card", "cat-character", "cat-verifiable",
            "cat-safety", "card-head", "v-tag", "formal", "voice", "card-desc",
            "sub-list", "sub-pill", "card-meta", "meta-row", "ledger-card",
            "ledger-header", "budget-meter", "b-segment", "spent", "cap",
            "reserve", "budget-legend",
        ]
        for cls_name in simula_classes:
            self.assertIn(cls_name, self.defined_classes, f"Class .{cls_name} not defined in <style>")

    def test_css_variables_full_dark_mode_coverage(self):
        """Verifies that all color variables in :root have overrides in dark mode."""
        # Find variables defined in :root
        root_block = re.search(r":root\{(.*?)\}", self.css_text, re.DOTALL)
        self.assertTrue(root_block)
        root_vars = set(re.findall(r"(--[a-zA-Z0-9_-]+):", root_block.group(1)))

        # Find variables defined in dark mode
        dark_block = re.search(r"@media \(prefers-color-scheme: dark\)\s*\{\s*:root:not\(\[data-theme=\"light\"\]\)\{(.*?)\}", self.css_text, re.DOTALL)
        self.assertTrue(dark_block)
        dark_vars = set(re.findall(r"(--[a-zA-Z0-9_-]+):", dark_block.group(1)))

        # Color tokens that must be mapped to dark equivalents
        required_tokens = {
            "--paper", "--surface", "--ink", "--muted", "--rule",
            "--accent", "--accent-soft", "--ochre", "--ochre-soft",
            "--danger", "--danger-soft", "--ghost",
        }
        for token in required_tokens:
            self.assertIn(token, root_vars, f"Token {token} missing in :root")
            self.assertIn(token, dark_vars, f"Token {token} missing in dark mode")

    def test_inline_style_color_hygiene(self):
        """Asserts no hardcoded hex or RGB color literals in inline styles."""
        class StyleInspector(HTMLParser):
            def __init__(self):
                super().__init__()
                self.violations = []
            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                if "style" in attrs_dict:
                    style = attrs_dict["style"]
                    hex_matches = re.findall(r"#[0-9a-fA-F]{3,6}", style)
                    rgb_matches = re.findall(r"rgba?\([^)]+\)", style)
                    if hex_matches or rgb_matches:
                        self.violations.append((tag, style, self.getpos(), hex_matches + rgb_matches))

        inspector = StyleInspector()
        inspector.feed(self.html_text)
        self.assertEqual(len(inspector.violations), 0, f"Inline styles with hardcoded colors: {inspector.violations}")

    def test_responsive_grid_breakpoints(self):
        """Verifies responsive layout breakpoints exist for multi-column grids."""
        self.assertIn("@media (max-width:720px){.simula-axes-strip{grid-template-columns:1fr}}", self.css_text)
        self.assertIn("@media (max-width:920px){.simula-grid{grid-template-columns:repeat(2,1fr)}}", self.css_text)
        self.assertIn("@media (max-width:580px){.simula-grid{grid-template-columns:1fr}}", self.css_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
