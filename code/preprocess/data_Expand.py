import xlrd
import sys
import xlswriter

reload(sys)
sys.setdefaultencoding('utf-8')

wb=xlrd.open_workbook('/Users/user/Desktop/research/search_new.xlsx')

store=wb.sheet_by_name(u'store_data')
genre=wb.sheet_by_name(u'genre')
test=wb.sheet_by_name(u'test')
l=62
length=1
count=0
kv={}          #kv -> all genres
for i in range(0,l):        #all
    str1=genre.cell(i,0).value
    kv[str1]=0

for j in range(1,90):        #compare
    str1=store.cell(j,3).value
    kv[str1]=1

for i,v in kv.items():
    if v==0:
        data=(i,0,0,0,0,0,0,'','','',0)
        str1='D'+str(length)
        test.write_row(str1,data)
        length=length+1
