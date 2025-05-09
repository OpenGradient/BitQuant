add:
	while IFS= read -r address; do [ "$address" = "STOP" ] && exit 0; curl -s 'https://agents-api.opengradient.ai/api/whitelist/add' -H "X-API-Key: ${WHITELIST_API_KEY}" -X POST --data "{\"address\":\"$address\"}" -H "Content-Type: application/json" > /dev/null; sleep 1; done < whitelist.txt

list:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}"

count:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}" | jq '.allowed | length'

add:
	curl 'https://agents-api.opengradient.ai/api/whitelist/add' -H "X-API-Key: ${WHITELIST_API_KEY}" -XPOST --data '{"address":""}' -H "Content-Type: application/json"
