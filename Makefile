add:
	uv add -r requirements.txt
	clear

run:
	adk web

git:
	git add .
	git status
	git commit -m "added file"
	git push
	clear