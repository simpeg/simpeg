import tomllib
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.resolve()
pyproject_file = root_dir / "pyproject.toml"

with open(pyproject_file, "rb") as f:
    pyproject = tomllib.load(f)

style_requirements = pyproject["project"]["optional-dependencies"]["style"]
for req in style_requirements:
    print(req)
