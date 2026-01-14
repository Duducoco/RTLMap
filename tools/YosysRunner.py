import subprocess
import os
import argparse
from pathlib import Path
from typing import List, Optional, Union

# Define paths for default usage
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DEFAULT_OSS_CAD_SUITE_PATH = PROJECT_ROOT / "oss-cad-suite"


class YosysRunner:
    def __init__(self, oss_cad_suite_path: Optional[Path] = None):
        self.oss_cad_suite_path = oss_cad_suite_path or DEFAULT_OSS_CAD_SUITE_PATH
        self.yosys_exe = self.oss_cad_suite_path / "bin" / "yosys.exe"
        self.env = self._setup_environment()

    def _setup_environment(self) -> dict:
        env = os.environ.copy()

        # YOSYSHQ_ROOT
        # The bat file uses %~dp0 which includes trailing backslash.
        yosys_root = str(self.oss_cad_suite_path) + os.sep
        env["YOSYSHQ_ROOT"] = yosys_root

        # SSL_CERT_FILE
        env["SSL_CERT_FILE"] = str(self.oss_cad_suite_path / "etc" / "cacert.pem")

        # PATH
        # %YOSYSHQ_ROOT%bin;%YOSYSHQ_ROOT%lib;%PATH%
        bin_path = str(self.oss_cad_suite_path / "bin")
        lib_path = str(self.oss_cad_suite_path / "lib")
        env["PATH"] = f"{bin_path};{lib_path};{env.get('PATH', '')}"

        # PYTHON_EXECUTABLE
        env["PYTHON_EXECUTABLE"] = str(self.oss_cad_suite_path / "lib" / "python3.exe")

        # QT variables
        env["QT_PLUGIN_PATH"] = str(self.oss_cad_suite_path / "lib" / "qt5" / "plugins")
        env["QT_LOGGING_RULES"] = "*=false"

        # GTK variables
        env["GTK_EXE_PREFIX"] = yosys_root
        env["GTK_DATA_PREFIX"] = yosys_root
        env["GDK_PIXBUF_MODULEDIR"] = str(
            self.oss_cad_suite_path / "lib" / "gdk-pixbuf-2.0" / "2.10.0" / "loaders"
        )
        env["GDK_PIXBUF_MODULE_FILE"] = str(
            self.oss_cad_suite_path
            / "lib"
            / "gdk-pixbuf-2.0"
            / "2.10.0"
            / "loaders.cache"
        )

        # OPENFPGALOADER_SOJ_DIR
        env["OPENFPGALOADER_SOJ_DIR"] = str(
            self.oss_cad_suite_path / "share" / "openFPGALoader"
        )

        return env

    def run(self, commands: List[str], output_file: Optional[str] = None) -> str:
        """
        Run Yosys with the given commands.

        Args:
            commands: List of Yosys commands to execute.
            output_file: Optional path to write the log output to.
        """
        # Join commands with semicolons
        command_str = "; ".join(commands)

        print(f"Executing Yosys commands: {command_str}")
        print("-" * 20)

        try:
            result = subprocess.run(
                [str(self.yosys_exe), "-p", command_str],
                env=self.env,
                capture_output=True,
                text=True,
            )

            if output_file:
                with open(output_file, "w") as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\nSTDERR:\n")
                        f.write(result.stderr)

            if result.returncode != 0:
                print("Yosys execution failed!")
                print(result.stderr)
                print(result.stdout)
                raise RuntimeError("Yosys execution failed")

            print("Yosys execution successful.")
            return result.stdout

        except Exception as e:
            print(f"An error occurred while running Yosys: {e}")
            raise

    def get_json(
        self,
        top_module: str,
        output_dir: Union[str, Path],
        flist: Optional[Union[str, Path]] = None,
        files: Optional[List[str]] = None,
    ) -> str:
        """
        Run Yosys to convert SystemVerilog to JSON using slang plugin.
        """
        if not flist and not files:
            raise ValueError("Either flist or files must be provided")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        plugin_command = ["plugin -i slang"]

        if flist:
            flist_path = Path(flist)
            read_command = [f"read_slang -f {flist_path} -top {top_module}"]
        else:
            # Join file paths with spaces
            files_str = " ".join(files)
            read_command = [f"read_slang {files_str} -top {top_module}"]

        pre_synth_command = [
            "flatten",
            "proc",
            "opt_clean",
            "opt_muxtree",
            "opt_reduce",
            "opt_merge",
            "opt_clean",
        ]
        write_command = [f"write_json {output_dir / f'{top_module}.json'}"]

        commands = plugin_command + read_command + pre_synth_command + write_command

        return self.run(commands)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Yosys synthesis")

    # Create a mutually exclusive group for input files
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--flist", help="Path to the file list (manifest.flist)")
    input_group.add_argument("--files", nargs="+", help="List of input files")

    parser.add_argument("--top", required=True, help="Top level module name")
    parser.add_argument(
        "--output-dir", default="RTLIL_json", help="Directory to save the output JSON"
    )

    args = parser.parse_args()

    args.output_dir = Path(args.flist).parent / args.output_dir if args.flist else Path(args.output_dir)

    runner = YosysRunner()

    try:
        runner.get_json(
            top_module=args.top,
            output_dir=args.output_dir,
            flist=args.flist,
            files=args.files,
        )
    except Exception as e:
        print(f"Failed to run Yosys: {e}")
