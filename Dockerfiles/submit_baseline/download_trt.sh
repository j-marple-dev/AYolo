#!/usr/bin/env bash

# Authorize to GitHub to get the latest release tar.gz
# Requires: oauth token, https://help.github.com/articles/creating-an-access-token-for-command-line-use/
# Requires: jq package to parse json

# Your oauth token goes here, see link above
OAUTH_TOKEN="de545234df337a702ac6cf2231d159074334462a"
# Repo owner (user id)
OWNER="j-marple-dev"
# Repo name
REPO="aigc-tr4-submit"
# The file name expected to download. This is deleted before curl pulls down a new one
FILE_NAME="nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb"

# Concatenate the values together for a 
API_URL="https://$OAUTH_TOKEN:@api.github.com/repos/$OWNER/$REPO"

# Gets info on latest release, gets first uploaded asset id of a release,
# More info on jq being used to parse json: https://stedolan.github.io/jq/tutorial/
ASSET_ID=$(curl $API_URL/releases/latest | jq -r '.assets[0].id')
echo "Asset ID: $ASSET_ID"

# curl does not allow overwriting file from -O, nuke
rm -f $FILE_NAME

# curl:
# -O: Use name provided from endpoint
# -J: "Content Disposition" header, in this case "attachment"
# -L: Follow links, we actually get forwarded in this request
# -H "Accept: application/octet-stream": Tells api we want to dl the full binary
curl -O -J -L -H "Accept: application/octet-stream" "$API_URL/releases/assets/$ASSET_ID"
