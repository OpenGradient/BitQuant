add:
	cat whitelist.json | jq -r '.allowed[]' | while read address; do curl http://localhost:5000/api/whitelist/add -H "X-API-Key: hello" -XPOST --data "{\"address\":\"$address\"}" -H "Content-Type: application/json"; sleep 0.5; done

list:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: hello"

