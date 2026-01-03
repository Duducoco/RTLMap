import os
import re
import json
import argparse
from pathlib import Path


def parse_html_files(directory):
    module_map = {}

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # Regex to find module name in title
    # Example: <title>Unified Coverage Report :: Module :: cv32e40p_core</title>
    # Example: <title>Unified Coverage Report :: Module Instance :: cv32e40p_wrapper.core_i.id_stage_i.controller_i</title>
    title_pattern = re.compile(
        r"<title>.*:: Module(?: Instance)? ::\s*([.\w]+)</title>"
    )

    # Alternative regex for body content if title fails
    # Example: <span class=titlename>Module : <a href="#"  onclick="showContent('tag_cv32e40p_core')">cv32e40p_core</a></span>
    # Example: <span class=titlename>Module Instance : <a href="hierarchy.html#tag_urg_inst_23" >cv32e40p_wrapper.core_i.id_stage_i.controller_i</a></span>
    span_pattern = re.compile(r"Module(?: Instance)?\s*:\s*<a[^>]*>([.\w]+)</a>")

    print(f"Scanning directory: {directory}")

    files = os.listdir(directory)
    html_files = [f for f in files if f.endswith(".html") and f.startswith("mod")]

    print(f"Found {len(html_files)} 'mod*.html' files.")

    for filename in html_files:
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

                match = title_pattern.search(content)
                if match:
                    module_name = match.group(1)
                    module_map[module_name] = filename
                else:
                    # Try alternative pattern
                    match = span_pattern.search(content)
                    if match:
                        module_name = match.group(1)
                        module_map[module_name] = filename
                    else:
                        print(f"Warning: Could not extract module name from {filename}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return module_map


def main():

    designs_root = Path(__file__).parent.absolute()
    # 查找designs_root中的文件夹
    design_folders = [f for f in designs_root.iterdir() if f.is_dir()]
    
    if not design_folders:
        print(f"No design folders found in {designs_root}")
        return
    for design_folder in design_folders:
        coverage_reports_dir = design_folder / "coverage_reports"
        if coverage_reports_dir.exists():
            coverage_reports = [cov for cov in coverage_reports_dir.iterdir() if cov.is_dir()]
            if coverage_reports:
                coverage_report_directory = coverage_reports[0]
                output_file = design_folder / "module2html.json"

                module_map = parse_html_files(coverage_report_directory)

                if module_map:
                    try:
                        with open(output_file, "w", encoding="utf-8") as f:
                            json.dump(module_map, f, indent=4)
                        print(
                            f"Successfully generated {output_file} with {len(module_map)} entries."
                        )
                    except Exception as e:
                        print(f"Error writing output file: {e}")
                else:
                    print("No modules found or map is empty.")
            else:
                print(f"No coverage report found in {coverage_reports_dir}")
        else:
            print(f"No coverage_reports directory in {design_folder}")


    # parser = argparse.ArgumentParser(
    #     description="Generate module2html.json from coverage report HTML files."
    # )
    # parser.add_argument("design_name", nargs="?", default=".", help="design_name")
    # parser.add_argument(
    #     "--output", "-o", help="Output JSON file path", default="module2html.json"
    # )

    # args = parser.parse_args()

    # directory = Path(args.design_name) / "coverage_reports" / "coverage_report1"
    # output_file = Path(args.design_name) / args.output

    # # If output file is just a filename, put it in the scanned directory by default if not specified otherwise
    # if os.path.dirname(output_file) == "" and directory != ".":
    #     output_file = os.path.join(directory, output_file)

    # module_map = parse_html_files(directory)

    # if module_map:
    #     try:
    #         with open(output_file, "w", encoding="utf-8") as f:
    #             json.dump(module_map, f, indent=4)
    #         print(
    #             f"Successfully generated {output_file} with {len(module_map)} entries."
    #         )
    #     except Exception as e:
    #         print(f"Error writing output file: {e}")
    # else:
    #     print("No modules found or map is empty.")


if __name__ == "__main__":
    main()
