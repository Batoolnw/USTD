#!/usr/bin/env python3
# Inspired by https://github.com/1337r00t/SnapMap/ & https://github.com/CaliAlec/snap-map-private-api/
# Created by sc1341 to be used to search the snapmap from a given address
#
#
import requests, time, json


def download_contents(data):
	i = 0
	l = len(data["manifest"]["elements"])
	print("Downloading " + str(l) + " media items")
	for value in data["manifest"]["elements"]:
	    idnum = value['id']
	    info = value['snapInfo']
	    info = value['snapInfo']
	    media = info.get('streamingMediaInfo')
	    preview_url = None
	    media_url = None
#	    overlay_url = None
	    if media:
	        if media.get('previewUrl'):
	            preview_url = media['prefixUrl'] + media['previewUrl']
	            with open('Media-Snap-Map/'+ str(idnum) + ".jpg", "wb") as f:
	                f.write(requests.get(preview_url).content)
	        if media.get('mediaUrl'):
	            media_url = media['prefixUrl'] + media['mediaUrl']
	            with open('Media-Snap-Map/'+ str(idnum) + ".mp4", "wb") as f:
	                f.write(requests.get(media_url).content)
#	        if media.get('overlayUrl'):
#	            overlay_url = media['prefixUrl'] + media['overlayUrl']
#	            with open('Media-Snap-Map/'+ str(idnum) + ".png", "wb") as f:
#	                f.write(requests.get(overlay_url).content)

#		filetype = value["snapInfo"]["streamingThumbnailInfo"]["infos"][0]["thumbnailUrl"].split(".")[-1]
#		with open('Media-Snap-Map/'+ str(i) + "." + filetype, "wb") as f:
#			f.write(requests.get(value["snapInfo"]["streamingThumbnailInfo"]["infos"][0]["thumbnailUrl"]).content)
	    i += 1
	    time.sleep(.5)


def export_json(data):
	filename = "snapmap_data.json"
	with open('Media-Snap-Map/'+ filename, "w") as f:
		f.write(json.dumps(data))
	print("Wrote JSON data to file" + filename)

def getEpoch():
	return requests.post('https://ms.sc-jpl.com/web/getLatestTileSet',headers={'Content-Type':'application/json'},data='{}').json()['tileSetInfos'][1]['id']['epoch']


def mainsnap():
#	args = parse_args()
#	os.mkdir(args.address + "-Snap-map")
#	os.chdir(args.address + "-Snap-map")
#	lat, lon = address_to_coordinates(args.address)
	lat = '24.696796'
	lon = '46.678700'
	post_data = '{"requestGeoPoint":{"lat":'+str(lat)+',"lon":'+str(lon)+'},"tileSetId":{"flavor":"default","epoch":'+str(getEpoch())+',"type":1},"radiusMeters":5000}'
	r = requests.post("https://ms.sc-jpl.com/web/getPlaylist", headers={"Content-Type":"application/json","User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:77.0) Gecko/20100101 Firefox/77.0"},data=post_data)
	export_json(r.json())
	download_contents(r.json())





