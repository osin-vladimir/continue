install_root:
	npm install

clean:
	find . \( -name "node_modules" -o -name "dist" \) -type d -prune -exec rm -rf '{}' +