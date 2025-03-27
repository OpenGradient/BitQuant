add:
	cat whitelist.json | jq -r '.allowed[]' | while read address; do curl 'https://agents-api.opengradient.ai/api/whitelist/add' -H "X-API-Key: ${WHITELIST_API_KEY}" -XPOST --data "{\"address\":\"$address\"}" -H "Content-Type: application/json"; sleep 1; done

list:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}"

count:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}" | jq '.allowed | length'

