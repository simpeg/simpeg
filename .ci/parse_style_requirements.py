import toml
import pathlib

root_dir = pathlib.Path(__file__).parent.parent.resolve()
pyproject_file = root_dir / "pyproject.toml"

pyproject = toml.load(pyproject_file)
style_requirements = pyproject["project"]["optional-dependencies"]["style"]
for req in style_requirements:
    print(req)
