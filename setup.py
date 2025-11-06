import setuptools
# --- IGNORE ---
with open("README.md", "r", encoding="utf-8") as fh:
    long_discription = fh.read()

__version__ = "0.0.0"

repo_name = "TextSummarizer"
author_user_name = "Aakash556-AI"
src_repo = "TextSummarizer"
author_email = "kumarravishek36@gmail.com"

setuptools.setup(
    name=repo_name,
    version=__version__,
    author=author_user_name,
    author_email=author_email,
    description="A small python package for NLP App",
    long_description=long_discription,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{author_user_name}/{src_repo}",
    project_urls={
        "Bug Tracker": f"https://github.com/{author_user_name}/{src_repo}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
