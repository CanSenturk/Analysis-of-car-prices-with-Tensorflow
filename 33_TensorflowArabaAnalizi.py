#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn


# In[4]:


dataFrame=pd.read_excel("merc.xlsx")


# In[5]:


dataFrame.head()


# In[6]:


dataFrame.describe()


# In[7]:


dataFrame.isnull().sum() #.sum kaçtane kolonda NaN var söyler


# In[8]:


plt.figure(figsize=(7,5))
sbn.displot(dataFrame["price"])#dağılım grafiği çizer seabornda


# In[9]:


sbn.countplot(dataFrame["year"])


# In[10]:


dataFrame.corr()


# In[11]:


dataFrame.corr()["price"].sort_values()#price sütununun korelasyonunu getirdi .sort_values ile küçükten büyüğe doğru sıraladı değerleri


# In[12]:


sbn.scatterplot(x="mileage",y="price",data=dataFrame)#nokta nokta verileri çizdir demek x özelliklerden birini seçiyoruz y ye ise de gitmek istediğimiz veriyi


# In[13]:


sbn.scatterplot(x="tax",y="price",data=dataFrame)#burada vergi fiyatlarının araba fiyatlarına olan etkisini analiz ettik.


# In[14]:


sbn.scatterplot(x="engineSize",y="price",data=dataFrame)#motor gücünün araba fiyatlarına olan etkisini analiz ettik


# In[15]:


sbn.scatterplot(x="year",y="price",data=dataFrame)#yılın araba fiyatları üzerindeki analizi


# In[16]:


dataFrame.sort_values("price",ascending=False).head(20)#en pahalı 20 arabayı getirdi. ascending=false dersek en pahalıdan başlar azalarak iner false true farkı budur


# In[17]:


dataFrame.sort_values("price",ascending=True).head(20)#burda fiyatı yükselen bir şekilde dizdi


# In[18]:


len(dataFrame)#len methodu kaçtane verimiz olduğunu gösterir


# In[19]:


len(dataFrame)*0.01#verilerin yüzde 1'ini aldık


# In[20]:


yuzdeDoksanDokuzDf=dataFrame.sort_values("price",ascending=False).iloc[131:]#fiyatı en yüksek 131 araba atıldı ve yuzdeDoksanDokuzDf değişkenine tanımlandı.


# In[21]:


yuzdeDoksanDokuzDf


# In[22]:


yuzdeDoksanDokuzDf.describe()


# In[23]:


plt.figure(figsize=(7,5))
sbn.distplot(yuzdeDoksanDokuzDf["price"])


# In[24]:


dataFrame.groupby("year").mean()["price"]#dataframe'i yıllara göre fiyatlarının ortalamasını aldı. mean()=ortalamasını alır


# In[25]:


yuzdeDoksanDokuzDf.groupby("year").mean()["price"]#burdada yuzdeDoksanDokuzDf değişkenin yıllara göre ortalama fiyatlarına baktık


# In[26]:


dataFrame[dataFrame.year!=1970].groupby("year").mean()["price"]#veri temizliği 1970 yılındaki veriyi attı


# In[27]:


dataFrame=yuzdeDoksanDokuzDf


# In[28]:


dataFrame.describe()


# In[29]:


dataFrame=dataFrame[dataFrame.year!=1970]#veri atma bu burda 1970 yılını attı


# In[30]:


dataFrame.groupby("year").mean()["price"] #1970 hiç görünmüyor


# In[31]:


dataFrame.head()


# In[32]:


dataFrame=dataFrame.drop("transmission",axis=1)# .drop düşür anlamına gelir burda transmission kolonu full düşer


# In[33]:


dataFrame#transmission yok


# In[34]:


y=dataFrame["price"].values # "y" bizim ulaşmak istediğimiz şey price yani fiyatlar .values dediğimizde numpy dizisine dönüşür.
x=dataFrame.drop("price",axis=1).values # "x" ise tablodaki geriye kalan değerler özellikler yani burda .drop diyerek price düşürdük geriye kalan tüm özellikleri "x"e eşitledik


# In[35]:


y


# In[36]:


x


# In[37]:


from sklearn.model_selection import train_test_split #bu ifade x ve y dizilerimizi eğiteceğimiz(train) ve test(test) edeceğimiz diziye bölüyordu.


# In[38]:


x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.30, random_state=10)#burada verilerin yüzde 30unu teste sokucaz. random_state= testen çıkan verilerin rastgele dağılmasını sağlar her defasında aynı sonucu verir.


# In[39]:


len(x_train)


# In[40]:


len(x_test)


# In[43]:


from sklearn.preprocessing import MinMaxScaler#ölçeklendirmek için kullanılır


# In[44]:


scaler=MinMaxScaler()#minimum maximum ölçeklendirme için kullanılır


# In[45]:


x_train=scaler.fit_transform(x_train)


# In[46]:


x_test=scaler.transform(x_test)


# In[47]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[49]:


x_train.shape#burda 9090 tane verimiz 5 tane kolonda özellikte analamına geliyor


# In[50]:


model=Sequential()#model oluşturuyoruz
model.add(Dense(12,activation="relu"))#katman ekliyoruz
model.add(Dense(12,activation="relu"))#katman ekliyoruz
model.add(Dense(12,activation="relu"))#katman ekliyoruz
model.add(Dense(12,activation="relu"))#katman ekliyoruz
model.add(Dense(1))#çıkış katmanı
model.compile(optimizer="adam",loss="mse")


# In[51]:


model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),batch_size=250,epochs=300)


# In[52]:


kayipVerisi=pd.DataFrame(model.history.history)


# In[53]:


kayipVerisi.head()


# In[54]:


kayipVerisi.plot()


# In[55]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[56]:


tahminDizisi=model.predict(x_test)


# In[57]:


tahminDizisi


# In[59]:


mean_absolute_error(y_test,tahminDizisi)


# In[60]:


dataFrame.describe()


# In[62]:


plt.scatter(y_test,tahminDizisi)
plt.plot(y_test,y_test,"g-*")


# In[64]:


dataFrame.iloc[2]


# In[67]:


yeniArabaSeries=dataFrame.drop("price",axis=1).iloc[2]


# In[68]:


yeniArabaSeries


# In[69]:


yeniArabaSeries=scaler.transform(yeniArabaSeries.values.reshape(-1,5))


# In[70]:


model.predict(yeniArabaSeries)


# In[ ]:




