all: venv install

venv:
	virtualenv .venv -p python3

install:
	echo "Installing packages from requirements.txt"
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/python manage.py runserver $(port) --settings=reports.settings.$(env) 

clean:
	rm *.pyc

requirements:
	.venv/bin/pip freeze > requirements.txt

manage:
	.venv/bin/python manage.py $(command) --settings=reports.settings.$(env)

