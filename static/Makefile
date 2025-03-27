add:
	total=$(cat whitelist.json | jq -r '.allowed[]' | wc -l); cat whitelist.json | jq -r '.allowed[]' | awk -v total="$total" '{print "Processing [" NR "/" total "]: " $0}' | while read line; do address=$(echo "$line" | sed 's/Processing \[[0-9]*\/[0-9]*\]: //'); echo "$line"; curl 'https://agents-api.opengradient.ai/api/whitelist/add' -H "X-API-Key: ${WHITELIST_API_KEY}" -XPOST --data "{\"address\":\"$address\"}" -H "Content-Type: application/json"; sleep 1; done

list:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}"

count:
	curl 'https://agents-api.opengradient.ai/api/whitelist' -H "X-API-Key: ${WHITELIST_API_KEY}" | jq '.allowed | length'

add:
	curl 'https://agents-api.opengradient.ai/api/whitelist/add' -H "X-API-Key: ${WHITELIST_API_KEY}" -XPOST --data '{"address":""}' -H "Content-Type: application/json"

check:
	curl 'https://agents-api.opengradient.ai/api/whitelisted?address=5PrqzSAWim3zHfDWoTVfGv85d7eTJGZBmrv4NVqoTTqt'