rm -r build
rm -r dist
rm -r modestga.egg-info
source venv/bin/activate
python setup.py sdist bdist_wheel
twine upload dist/*
