from setuptools import setup
from glob import glob

package_name = "formation_control"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("launch_folder/formation_launch_2_3.py")),
        ("share/" + package_name, glob("launch_folder/formation_launch_2_1.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [f"2_1_agent = {package_name}.the_agent_2_1:main",
                            f"2_3_agent = {package_name}.the_agent_2_3:main"],
    },
)
