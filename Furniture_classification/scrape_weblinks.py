from bs4 import BeautifulSoup
import os
import urllib.request
import time


from selenium.webdriver.common.keys import Keys
from selenium import webdriver


# Make folder structure for scraping data

def makedirs(path):
	if not os.path.exists(path):
		os.makedirs(path)
		print('{} Created'.format(path))

path_list = ['dataset/original/chairs', 'dataset/original/wardrobes', 'dataset/original/curtains','dataset/original/sofas']
web_links = ['https://www.ikea.com/in/en/search/products/?q=chairs', 'https://www.ikea.com/in/en/search/products/?q=wardrobes', 
				'https://www.ikea.com/in/en/search/products/?q=curtains','https://www.ikea.com/in/en/search/products/?q=sofas']


def download_images(image_url, save_dir):
	browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")

	browser.get(image_url)

	# Traverse page for 4 sections of 48 images each
	counter = 0
	try:
		while counter <= 3:
			browser.find_element_by_xpath("""//*[@id="content"]/div/div/div[3]/div[2]/div[3]/button""").click()
			time.sleep(5)
			counter += 1
	except:
		pass

	browser.refresh()

	# This will get the html after on-load javascript
	html = browser.execute_script("return document.documentElement.innerHTML;")
	soup = BeautifulSoup(html, 'lxml')
	link_dir = soup.find_all('img')

	counter = 0
	try:
		for items in link_dir:
			url = items.attrs['src'].split('?')[0]
			if url.lower().endswith('.jpg'):
				exists = os.path.isfile(os.path.join(save_dir, '{}'.format(url.split('/')[-1])))
				counter += 1
				# print('Web link:', url)
				if not exists:
					urllib.request.urlretrieve(url, os.path.join(save_dir, '{}'.format(url.split('/')[-1])))
					print('Downloaded {}'.format(url))
				else:
					print(' File {} already downloaded'.format(url.split('/')[-1]))
	except Exception:
		print('Exception:', Exception )


	print('Total Images for {}:'.format(save_dir.split('/')[-1]), counter)
	time.sleep(2)
	try:
		browser.close()
	except:
		pass



# Make folder structure for training dataset

for path in path_list:
	makedirs(path)


# Download images for 4 categories

for index in range(4):
	download_images(web_links[index], path_list[index])