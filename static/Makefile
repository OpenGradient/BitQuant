add:
	cat whitelist.json | jq -r '.allowed[]' | while read address; do curl http://localhost:5000/api/whitelist/add -H "X-API-Key: hello" -XPOST --data "{\"address\":\"$address\"}" -H "Content-Type: application/json"; done

list:
	curl http://localhost:5000/api/whitelist/list -H "X-API-Key: hello"

