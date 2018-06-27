import urllib.request
import os
import random
import pprint as ppr
from bs4 import BeautifulSoup

header = {}
header_key = 'User-Agent'
header_value = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
header[header_key] = header_value


def get_proxy_ip():  # 获取代理ip
    proxyip_list = []
    proxy_ip_url = 'http://www.xicidaili.com/wt/'
    for i in range(1, 2):
        try:
            url = proxy_ip_url + str(i)
            req = urllib.request.Request(url, headers=header)
            res = urllib.request.urlopen(req).read()
            soup = BeautifulSoup(res, "html.parser")
            ips = soup.findAll('tr')
            for x in range(1, len(ips)):
                ip = ips[x]
                tds = ip.findAll('td')
                ip_tmp = tds[1].contents[0] + ':' + tds[2].contents[0]
                proxyip_list.append(ip_tmp)
        except:
            continue
    return proxyip_list


# 打开url接口函数
def url_open(url):
    req = urllib.request.Request(url)
    req.add_header(header_key, header_value)
    ''''' 
    #使用代理模拟真人访问而不是代码访问 
    iplist = get_proxy_ip() 
    #ppr.pprint(iplist)  #for test 
    ip_port = random.choice(iplist) 
    proxy_support = urllib.request.ProxyHandler({'http':ip_port}) 
    print('ip:port    ' + ip_port)  #for test 
    opener = urllib.request.build_opener(proxy_support) 
    opener.addheaders = [('User-Agent', 'Chrome/55.0.2883.87')] 
    urllib.request.install_opener(opener) 
    '''
    res = urllib.request.urlopen(url)
    html = res.read()

    print(url)
    return html


def get_page(url):
    html = url_open(url).decode('utf-8')

    a = html.find('current-comment-page') + 23
    b = html.find(']', a)  # 从a位置开始找到一个]符号
    print(html[a:b])  # for test

    return html[a:b]


def find_imgs(url):
    html = url_open(url).decode('utf-8')
    img_addrs = []

    a = html.find('img src=')  # 找到 img src= 的位置

    while a != -1:
        b = html.find('.jpg', a, a + 255)  # 找到从a位置开始，以 .jpg 结尾的地方
        if b != -1:  # find找不到时返回-1
            img_addrs.append('http:' + html[a + 9:b + 4])  # 图片链接地址追加到列表中, 9=len('img src="'), 4=len('.jpg')
        else:
            b = a + 9

        a = html.find('img src=', b)  # 下一次循环所找的位置就是从b开始

    # for each in img_addrs:
    #    print(each)

    return img_addrs


def save_imgs(folder, img_addrs):
    for each in img_addrs:
        filename = each.split('/')[-1]  # split以/分割字符串，-1取最后一个元素
        with open(filename, 'wb') as f:
            img = url_open(each)
            f.write(img)


def download_mm(folder='mm_dir', pages=25):
    if os.path.exists(folder):
        os.chdir(folder)
    else:
        os.mkdir(folder)
        os.chdir(folder)

    url = 'http://jandan.net/ooxx/'  # 实际图源来源于新浪服务器
    page_num = int(get_page(url))  # 函数get_page()

    for i in range(pages):
        page_num -= i
        page_url = url + 'page-' + str(page_num) + '#comments'
        img_addrs = find_imgs(page_url)  # 函数find_imgs()
        save_imgs(folder, img_addrs)


if __name__ == '__main__':
    download_mm()
