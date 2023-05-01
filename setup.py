import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
	long_description = fh.read()

print(setuptools.find_packages(where="src"))

setuptools.setup(
	name="zadu",
	version="0.0.8",
	author="Hyeon Jeon",
	author_email="hj@hcil.snu.ac.kr",
	description="A Python Toolkit for Evaluating the Reliability of Dimensionality Reduction Embeddings",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/hj-n/zadu",
	classifiers=[
			"Programming Language :: Python :: 3",
			"License :: OSI Approved :: MIT License",
			"Operating System :: OS Independent",
	],
	install_requires=[
		"numpy",
		"scikit-learn",
		"scipy",
		"snc",
		"matplotlib",
		"faiss-cpu"
	],
	package_dir={"": "src"},
	packages=["zadu", "zaduvis", "zadu.measures", "zadu.measures.utils"],
	python_requires=">=3.9.0",
)