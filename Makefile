create_folder:
	@mkdir /.opt

development: 
	@sudo cp -r public/ files/ /.opt/ 
	@sudo chown ${USER}:${USER} /.opt/files /.opt/public

run:
	@python3 inference.py