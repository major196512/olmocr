import unittest

from olmocr.bench.tests import (
    parse_html_tables,
    parse_markdown_tables,
)


class TestParseHtmlTables(unittest.TestCase):
    def test_basic_table(self):
        data = parse_html_tables("""
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
                        </tbody></table>""")[0]
        
        print(data)
        
        self.assertEqual(data.cell_text[0,0], "")
        self.assertEqual(data.cell_text[0,1], "ArXiv")

        self.assertEqual(data.left_relations[0,0], set())
        self.assertEqual(data.up_relations[0,0], set())

        self.assertEqual(data.left_relations[0,1], {(0,0)})
        self.assertEqual(data.up_relations[1,0], {(0,0)})

        self.assertEqual(data.heading_cells, {
            (0,0), (0,1), (0,2), (0,3),(0,4), (0,5),(0,6), (0,7), (0,8), (0,9)
        })

        self.assertEqual(data.top_heading_relations(1,3), {(0,3)})
        
        # If there are no left headings defined, then the left most column is considered the left heading
        print(data.left_heading_relations(1,3))
        self.assertEqual(data.left_heading_relations(1,3), {(1,0)})

    def test_multiple_top_headings(self):
        data = parse_html_tables("""
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
                        </tbody></table>""")[0]
        
        print(data)
        
        self.assertEqual(data.cell_text[0,0], "Fruit Costs in Unittest land")
        self.assertEqual(data.cell_text[1,0], "Fruit Type")
        self.assertEqual(data.cell_text[1,1], "Cost")
        self.assertEqual(data.cell_text[2,0], "Apples")
        self.assertEqual(data.cell_text[2,1], "$1.00")
        self.assertEqual(data.cell_text[3,0], "Oranges")
        self.assertEqual(data.cell_text[3,1], "$2.00")


        self.assertEqual(data.up_relations[1,0], {(0,0)})
        self.assertEqual(data.up_relations[1,1], {(0,0)})

        self.assertEqual(data.up_relations[2,0], {(1,0)})
        self.assertEqual(data.up_relations[2,1], {(1,1)})

        self.assertEqual(data.top_heading_relations(1,0), {(0,0)})
        self.assertEqual(data.top_heading_relations(1,1), {(0,0)})

        self.assertEqual(data.top_heading_relations(2,0), {(0,0), (1,0)})
        self.assertEqual(data.top_heading_relations(2,1), {(0,0), (1,1)})

    def test_4x4_table_with_spans(self):
        """Test a 4x4 table with various row spans and column spans"""
        data = parse_html_tables("""
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
                    </table>""")[0]

        print(data)

        # Test header row
        self.assertEqual(data.cell_text[0,0], "Header 1")
        self.assertEqual(data.cell_text[0,1], "Header 2-3")

        self.assertNotIn((0,2), data.cell_text)  # colspan=2, so that next cell is empty
        self.assertEqual(data.cell_text[0,3], "Header 4")

        # Test first body row
        self.assertEqual(data.cell_text[1,0], "Cell A (spans 2 rows)")
        self.assertEqual(data.cell_text[1,1], "Cell B")
        self.assertEqual(data.cell_text[1,2], "Cell C")
        self.assertEqual(data.cell_text[1,3], "Cell D (spans 2 rows)")

        # Test second body row
        self.assertNotIn((2,0), data.cell_text)
        self.assertEqual(data.cell_text[2,1], "Cell E-F (spans 2 cols)")

        # Test third body row
        self.assertEqual(data.cell_text[3,0], "Cell G")
        self.assertEqual(data.cell_text[3,1], "Cell H-I-J (spans 3 cols)")

        # Test heading cells
        self.assertEqual(data.heading_cells, {
            (0,0), (0,1), (0,3)
        })
