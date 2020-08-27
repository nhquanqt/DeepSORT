from google_drive_downloader import GoogleDriveDownloader as gdd
gdd.download_file_from_google_drive(file_id='1dwJJin4TmJ5zHjEu2iqRwRYRynBujpmX',
                                    dest_path='./instances_train2017.json',
                                    unzip=False)
import json
import requests
with open('instances_train2017.json') as json_file:
    itemData = json.load(json_file)
itemData.get('annotations')[0].get('category_id')
GroupId = [2,3,4,6,8]
NewJson = dict()
import os 
# Directory 
directory = "Images"
  
# Parent Directory path 
parent_dir = "./content"
  
# Path 
path = os.path.join(parent_dir, directory)
imageID = []
if not os.path.exists(path):
    os.makedirs(path)
for i in range(len(itemData.get('annotations'))):
    if itemData.get('annotations')[i].get('category_id') in GroupId:
      x = dict()
      x['bbox'] = itemData.get('annotations')[i].get('bbox')
      if itemData.get('annotations')[i].get('category_id') in [2,4]:
        x['category_id'] = 'group01'
      elif itemData.get('annotations')[i].get('category_id') in [6]:
        x['category_id'] = 'group03'
      elif itemData.get('annotations')[i].get('category_id') in [3]:
        x['category_id'] = 'group02'
      elif itemData.get('annotations')[i].get('category_id') in [8]:
        x['category_id'] = 'group04'
      x['image_id'] = itemData.get('annotations')[i].get('image_id')
      
      x['id'] = itemData.get('annotations')[i].get('id')
      x['area'] = itemData.get('annotations')[i].get('area')
      NewJson[i] = x
      imageID.append(x['image_id'])
with open('data.json', 'w') as f:
    json.dump(NewJson, f)
#get image by http://images.cocodataset.org/train2017/000000 + id .jpg   
for i in range(len(imageID)):
  response = requests.get("http://images.cocodataset.org/train2017/"+ "0"*(12-len(str(x['image_id']))) + str(x['image_id']))
  with open(os.path.join(path,str(x['image_id']) + '.png'), "wb") as image:
        image.write(response.content)
        image.close()