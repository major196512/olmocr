import unittest

from olmocr.bench.tests import (
    parse_html_tables,
    parse_markdown_tables,
)


class TestParseHtmlTables(unittest.TestCase):
    def test_basic_table(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th></th>
                                <th>ArXiv</th>
                                <th>Old<br>scans<br>math</th>
                                <th>Tables</th>
                                <th>Old<br>scans</th>
                                <th>Headers<br>&<br>footers</th>
                                <th>Multi<br>column</th>
                                <th>Long<br>tiny<br>text</th>
                                <th>Base</th>
                                <th>Overall</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Mistral OCR API</td>
                                <td>77.2</td>
                                <td>67.5</td>
                                <td>60.6</td>
                                <td>29.3</td>
                                <td>93.6</td>
                                <td>71.3</td>
                                <td>77.1</td>
                                <td>99.4</td>
                                <td>72.0±1.1</td>
                            </tr>
                        </tbody></table>"""
        )[0]

        print(data)

        self.assertEqual(data.cell_text[0, 0], "")
        self.assertEqual(data.cell_text[0, 1], "ArXiv")

        self.assertEqual(data.left_relations[0, 0], set())
        self.assertEqual(data.up_relations[0, 0], set())

        self.assertEqual(data.left_relations[0, 1], {(0, 0)})
        self.assertEqual(data.up_relations[1, 0], {(0, 0)})

        self.assertEqual(data.heading_cells, {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)})

        self.assertEqual(data.top_heading_relations(1, 3), {(0, 3)})

        # If there are no left headings defined, then the left most column is considered the left heading
        print(data.left_heading_relations(1, 3))
        self.assertEqual(data.left_heading_relations(1, 3), {(1, 0)})

    def test_multiple_top_headings(self):
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                               <th colspan="2">Fruit Costs in Unittest land</th>
                            </tr>
                            <tr>
                                <th>Fruit Type</th>
                                <th>Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Apples</td>
                                <td>$1.00</td>
                            </tr>
                            <tr>
                                <td>Oranges</td>
                                <td>$2.00</td>
                            </tr>                                 
                        </tbody></table>"""
        )[0]

        print(data)

        self.assertEqual(data.cell_text[0, 0], "Fruit Costs in Unittest land")
        self.assertEqual(data.cell_text[1, 0], "Fruit Type")
        self.assertEqual(data.cell_text[1, 1], "Cost")
        self.assertEqual(data.cell_text[2, 0], "Apples")
        self.assertEqual(data.cell_text[2, 1], "$1.00")
        self.assertEqual(data.cell_text[3, 0], "Oranges")
        self.assertEqual(data.cell_text[3, 1], "$2.00")

        self.assertEqual(data.up_relations[1, 0], {(0, 0)})
        self.assertEqual(data.up_relations[1, 1], {(0, 0)})

        self.assertEqual(data.up_relations[2, 0], {(1, 0)})
        self.assertEqual(data.up_relations[2, 1], {(1, 1)})

        self.assertEqual(data.top_heading_relations(1, 0), {(0, 0)})
        self.assertEqual(data.top_heading_relations(1, 1), {(0, 0)})

        self.assertEqual(data.top_heading_relations(2, 0), {(0, 0), (1, 0)})
        self.assertEqual(data.top_heading_relations(2, 1), {(0, 0), (1, 1)})

    def test_4x4_table_with_spans(self):
        """Test a 4x4 table with various row spans and column spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th>Header 1</th>
                                <th colspan="2">Header 2-3</th>
                                <th>Header 4</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2">Cell A (spans 2 rows)</td>
                                <td>Cell B</td>
                                <td>Cell C</td>
                                <td rowspan="2">Cell D (spans 2 rows)</td>
                            </tr>
                            <tr>
                                <td colspan="2">Cell E-F (spans 2 cols)</td>
                            </tr>
                            <tr>
                                <td>Cell G</td>
                                <td colspan="3">Cell H-I-J (spans 3 cols)</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print(data)

        # Test header row
        self.assertEqual(data.cell_text[0, 0], "Header 1")
        self.assertEqual(data.cell_text[0, 1], "Header 2-3")

        self.assertNotIn((0, 2), data.cell_text)  # colspan=2, so that next cell is empty
        self.assertEqual(data.cell_text[0, 3], "Header 4")

        # Test first body row
        self.assertEqual(data.cell_text[1, 0], "Cell A (spans 2 rows)")
        self.assertEqual(data.cell_text[1, 1], "Cell B")
        self.assertEqual(data.cell_text[1, 2], "Cell C")
        self.assertEqual(data.cell_text[1, 3], "Cell D (spans 2 rows)")

        # Test second body row
        self.assertNotIn((2, 0), data.cell_text)
        self.assertEqual(data.cell_text[2, 1], "Cell E-F (spans 2 cols)")

        # Test third body row
        self.assertEqual(data.cell_text[3, 0], "Cell G")
        self.assertEqual(data.cell_text[3, 1], "Cell H-I-J (spans 3 cols)")

        # Test heading cells
        self.assertEqual(data.heading_cells, {(0, 0), (0, 1), (0, 3)})

        self.assertEqual(data.left_heading_relations(0, 0), set())
        self.assertEqual(data.left_heading_relations(1, 0), set())
        self.assertEqual(data.left_heading_relations(2, 0), set())
        self.assertEqual(data.left_heading_relations(3, 0), set())

        self.assertEqual(data.top_heading_relations(0, 0), set())
        self.assertEqual(data.top_heading_relations(0, 1), set())
        self.assertEqual(data.top_heading_relations(0, 2), set())
        self.assertEqual(data.top_heading_relations(0, 3), set())

        self.assertEqual(data.left_heading_relations(1, 1), {(1, 0)})
        self.assertEqual(data.left_heading_relations(1, 2), {(1, 0)})
        self.assertEqual(data.left_heading_relations(1, 3), {(1, 0)})

        self.assertEqual(data.top_heading_relations(3, 1), {(0, 1), (0, 3)})

    def test_complex_multi_level_headers(self):
        """Test a table with multiple levels of headers and complex spanning"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="3"></th>
                                <th colspan="6">Main Category</th>
                            </tr>
                            <tr>
                                <th colspan="3">Sub Category A</th>
                                <th colspan="3">Sub Category B</th>
                            </tr>
                            <tr>
                                <th>A1</th>
                                <th>A2</th>
                                <th>A3</th>
                                <th>B1</th>
                                <th>B2</th>
                                <th>B3</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Row 1</td>
                                <td>10</td>
                                <td>20</td>
                                <td>30</td>
                                <td>40</td>
                                <td>50</td>
                                <td>60</td>
                            </tr>
                            <tr>
                                <td>Row 2</td>
                                <td>15</td>
                                <td>25</td>
                                <td>35</td>
                                <td>45</td>
                                <td>55</td>
                                <td>65</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Complex Multi-Level Headers Test ===")
        print(data)

        # Test the three-level header structure
        self.assertEqual(data.cell_text[0, 0], "")  # Empty corner cell
        self.assertEqual(data.cell_text[0, 1], "Main Category")

        self.assertEqual(data.cell_text[1, 1], "Sub Category A")
        self.assertEqual(data.cell_text[1, 4], "Sub Category B")

        self.assertEqual(data.cell_text[2, 1], "A1")
        self.assertEqual(data.cell_text[2, 2], "A2")
        self.assertEqual(data.cell_text[2, 3], "A3")
        self.assertEqual(data.cell_text[2, 4], "B1")
        self.assertEqual(data.cell_text[2, 5], "B2")
        self.assertEqual(data.cell_text[2, 6], "B3")

        # Test data rows
        self.assertEqual(data.cell_text[3, 0], "Row 1")
        self.assertEqual(data.cell_text[3, 1], "10")
        self.assertEqual(data.cell_text[4, 0], "Row 2")
        self.assertEqual(data.cell_text[4, 1], "15")

        # Test heading cells - all header rows should be marked as heading cells
        expected_heading_cells = {
            (0, 0), (0, 1),  # First header row
            (1, 1), (1, 4),  # Second header row
            (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)  # Third header row
        }
        self.assertEqual(data.heading_cells, expected_heading_cells)

        # Test top heading relations for data cells
        # Cell (3, 1) should have relations to all three levels of headers above it
        top_relations_3_1 = data.top_heading_relations(3, 1)
        self.assertIn((0, 1), top_relations_3_1)  # Main Category
        self.assertIn((1, 1), top_relations_3_1)  # Sub Category A
        self.assertIn((2, 1), top_relations_3_1)  # A1

        # Cell (3, 4) should relate to headers in the B column
        top_relations_3_4 = data.top_heading_relations(3, 4)
        self.assertIn((0, 1), top_relations_3_4)  # Main Category
        self.assertIn((1, 4), top_relations_3_4)  # Sub Category B
        self.assertIn((2, 4), top_relations_3_4)  # B1

        # Test left heading relations
        self.assertEqual(data.left_heading_relations(3, 1), {(3, 0)})  # Row 1
        self.assertEqual(data.left_heading_relations(4, 1), {(4, 0)})  # Row 2

    def test_left_headers_with_row_spans(self):
        """Test a table with left-side headers that have row spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="2" colspan="2"></th>
                                <th colspan="3">Quarter 1</th>
                                <th colspan="3">Quarter 2</th>
                            </tr>
                            <tr>
                                <th>Jan</th>
                                <th>Feb</th>
                                <th>Mar</th>
                                <th>Apr</th>
                                <th>May</th>
                                <th>Jun</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <th rowspan="3">North</th>
                                <th>Sales</th>
                                <td>100</td>
                                <td>110</td>
                                <td>120</td>
                                <td>130</td>
                                <td>140</td>
                                <td>150</td>
                            </tr>
                            <tr>
                                <th>Cost</th>
                                <td>50</td>
                                <td>55</td>
                                <td>60</td>
                                <td>65</td>
                                <td>70</td>
                                <td>75</td>
                            </tr>
                            <tr>
                                <th>Profit</th>
                                <td>50</td>
                                <td>55</td>
                                <td>60</td>
                                <td>65</td>
                                <td>70</td>
                                <td>75</td>
                            </tr>
                            <tr>
                                <th rowspan="3">South</th>
                                <th>Sales</th>
                                <td>200</td>
                                <td>210</td>
                                <td>220</td>
                                <td>230</td>
                                <td>240</td>
                                <td>250</td>
                            </tr>
                            <tr>
                                <th>Cost</th>
                                <td>100</td>
                                <td>105</td>
                                <td>110</td>
                                <td>115</td>
                                <td>120</td>
                                <td>125</td>
                            </tr>
                            <tr>
                                <th>Profit</th>
                                <td>100</td>
                                <td>105</td>
                                <td>110</td>
                                <td>115</td>
                                <td>120</td>
                                <td>125</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Left Headers with Row Spans Test ===")
        print(data)

        # Test top headers
        self.assertEqual(data.cell_text[0, 2], "Quarter 1")
        self.assertEqual(data.cell_text[0, 5], "Quarter 2")
        self.assertEqual(data.cell_text[1, 2], "Jan")
        self.assertEqual(data.cell_text[1, 3], "Feb")
        self.assertEqual(data.cell_text[1, 4], "Mar")

        # Test left headers with row spans
        self.assertEqual(data.cell_text[2, 0], "North")
        self.assertEqual(data.cell_text[2, 1], "Sales")
        self.assertEqual(data.cell_text[3, 1], "Cost")
        self.assertEqual(data.cell_text[4, 1], "Profit")

        self.assertEqual(data.cell_text[5, 0], "South")
        self.assertEqual(data.cell_text[5, 1], "Sales")

        # Test data values
        self.assertEqual(data.cell_text[2, 2], "100")
        self.assertEqual(data.cell_text[3, 2], "50")
        self.assertEqual(data.cell_text[5, 2], "200")

        # Test heading cells - should include both top headers and left headers
        # Top headers: rows 0-1, Left headers: column 0-1 in data rows
        self.assertIn((0, 2), data.heading_cells)  # Quarter 1
        self.assertIn((1, 2), data.heading_cells)  # Jan
        self.assertIn((2, 0), data.heading_cells)  # North (left header)
        self.assertIn((2, 1), data.heading_cells)  # Sales (left header)
        self.assertIn((5, 0), data.heading_cells)  # South (left header)

        # Test left heading relations with multiple levels
        # Data cell (2, 2) should have both North and Sales as left headers
        left_relations_2_2 = data.left_heading_relations(2, 2)
        self.assertIn((2, 0), left_relations_2_2)  # North
        self.assertIn((2, 1), left_relations_2_2)  # Sales

        # Data cell (3, 2) should have North (spans from row 2) and Cost as left headers
        left_relations_3_2 = data.left_heading_relations(3, 2)
        self.assertIn((2, 0), left_relations_3_2)  # North (row span)
        self.assertIn((3, 1), left_relations_3_2)  # Cost

        # Data cell (5, 3) should have South and Sales as left headers
        left_relations_5_3 = data.left_heading_relations(5, 3)
        self.assertIn((5, 0), left_relations_5_3)  # South
        self.assertIn((5, 1), left_relations_5_3)  # Sales

        # Test top heading relations
        top_relations_2_2 = data.top_heading_relations(2, 2)
        self.assertIn((0, 2), top_relations_2_2)  # Quarter 1
        self.assertIn((1, 2), top_relations_2_2)  # Jan

    def test_nested_header_groups_with_col_spans(self):
        """Test a complex table with nested header groups and various column spans"""
        data = parse_html_tables(
            """
                    <table border="1">
                        <thead>
                            <tr>
                                <th rowspan="4">Region</th>
                                <th rowspan="4">Store</th>
                                <th colspan="12">2024 Sales Data</th>
                            </tr>
                            <tr>
                                <th colspan="6">First Half</th>
                                <th colspan="6">Second Half</th>
                            </tr>
                            <tr>
                                <th colspan="3">Q1</th>
                                <th colspan="3">Q2</th>
                                <th colspan="3">Q3</th>
                                <th colspan="3">Q4</th>
                            </tr>
                            <tr>
                                <th>Jan</th>
                                <th>Feb</th>
                                <th>Mar</th>
                                <th>Apr</th>
                                <th>May</th>
                                <th>Jun</th>
                                <th>Jul</th>
                                <th>Aug</th>
                                <th>Sep</th>
                                <th>Oct</th>
                                <th>Nov</th>
                                <th>Dec</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td rowspan="2">West</td>
                                <td>Store A</td>
                                <td>10</td>
                                <td>11</td>
                                <td>12</td>
                                <td>13</td>
                                <td>14</td>
                                <td>15</td>
                                <td>16</td>
                                <td>17</td>
                                <td>18</td>
                                <td>19</td>
                                <td>20</td>
                                <td>21</td>
                            </tr>
                            <tr>
                                <td>Store B</td>
                                <td>22</td>
                                <td>23</td>
                                <td>24</td>
                                <td>25</td>
                                <td>26</td>
                                <td>27</td>
                                <td>28</td>
                                <td>29</td>
                                <td>30</td>
                                <td>31</td>
                                <td>32</td>
                                <td>33</td>
                            </tr>
                        </tbody>
                    </table>"""
        )[0]

        print("\n=== Nested Header Groups with Col Spans Test ===")
        print(data)

        # Test nested header structure
        self.assertEqual(data.cell_text[0, 0], "Region")
        self.assertEqual(data.cell_text[0, 1], "Store")
        self.assertEqual(data.cell_text[0, 2], "2024 Sales Data")

        self.assertEqual(data.cell_text[1, 2], "First Half")
        self.assertEqual(data.cell_text[1, 8], "Second Half")

        self.assertEqual(data.cell_text[2, 2], "Q1")
        self.assertEqual(data.cell_text[2, 5], "Q2")
        self.assertEqual(data.cell_text[2, 8], "Q3")
        self.assertEqual(data.cell_text[2, 11], "Q4")

        self.assertEqual(data.cell_text[3, 2], "Jan")
        self.assertEqual(data.cell_text[3, 7], "Jun")
        self.assertEqual(data.cell_text[3, 13], "Dec")

        # Test data rows
        self.assertEqual(data.cell_text[4, 0], "West")
        self.assertEqual(data.cell_text[4, 1], "Store A")
        self.assertEqual(data.cell_text[4, 2], "10")
        self.assertEqual(data.cell_text[5, 1], "Store B")
        self.assertEqual(data.cell_text[5, 2], "22")

        # Test all header cells are marked
        self.assertIn((0, 0), data.heading_cells)  # Region
        self.assertIn((0, 1), data.heading_cells)  # Store
        self.assertIn((0, 2), data.heading_cells)  # 2024 Sales Data
        self.assertIn((1, 2), data.heading_cells)  # First Half
        self.assertIn((2, 2), data.heading_cells)  # Q1
        self.assertIn((3, 2), data.heading_cells)  # Jan

        # Test multiple top heading relations for a data cell
        # Cell (4, 2) - January data for Store A should have all 4 levels of headers
        top_relations_4_2 = data.top_heading_relations(4, 2)
        self.assertIn((0, 2), top_relations_4_2)  # 2024 Sales Data
        self.assertIn((1, 2), top_relations_4_2)  # First Half
        self.assertIn((2, 2), top_relations_4_2)  # Q1
        self.assertIn((3, 2), top_relations_4_2)  # Jan

        # Cell (4, 7) - June data should relate to Q2 and First Half
        top_relations_4_7 = data.top_heading_relations(4, 7)
        self.assertIn((0, 2), top_relations_4_7)  # 2024 Sales Data
        self.assertIn((1, 2), top_relations_4_7)  # First Half
        self.assertIn((2, 5), top_relations_4_7)  # Q2
        self.assertIn((3, 7), top_relations_4_7)  # Jun

        # Cell (4, 13) - December data should relate to Q4 and Second Half
        top_relations_4_13 = data.top_heading_relations(4, 13)
        self.assertIn((0, 2), top_relations_4_13)  # 2024 Sales Data
        self.assertIn((1, 8), top_relations_4_13)  # Second Half
        self.assertIn((2, 11), top_relations_4_13)  # Q4
        self.assertIn((3, 13), top_relations_4_13)  # Dec

        # Test left heading relations
        # Store B row that says "22" should relate to just West
        left_relations_5_2 = data.left_heading_relations(5, 2)
        self.assertEqual(data.cell_text[5,2], "22")
        self.assertIn((4, 0), left_relations_5_2)
        self.assertEqual(len(left_relations_5_2), 1)

        # But of the left headings themselves at the top, January, should have both Region and Store
        self.assertEqual(data.left_heading_relations(3,2), {(0, 0), (0, 1)})