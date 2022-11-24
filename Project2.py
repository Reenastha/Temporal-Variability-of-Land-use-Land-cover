#!/usr/bin/env python
# coding: utf-8

# <center> <h1> <font color='magenta'> Temporal Variability of Land use Land cover <font> </h1> </center>

# <h3><center><font color='magenta'> Team Magenta </font></center></h3>

# <h4><font color='red'> Objective of Project </font> </h4>

# <p4> 1. Distributed area of 5 land use variables of Jefferson county </p4>
# 
# <p4> 2. Comparison of 5 land use variables of SETX in 2016 and 2019 </p4>
# 
# <p4> 3. Percentage of area 5 land use variables of SETX </p4>
# 
# <p4> 4. Change in developed land over the 18 years in SETX </p4>
# 
# <p4> 5. Change in forest area over 18 years in SETX </p4>

# <h4><font color='red'> Load Libraries </font> </h4>

# In[1]:


from osgeo import gdal
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import os
import rasterio as rs
from rasterio.plot import show


# <h4><font color='red'> Set Working Directory for Texas County Map </font> </h4>

# In[2]:


os.chdir("C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/project2/Texas Counties Map")
county = gpd.read_file('geo_export_3f7968bd-97ea-4376-864b-379405bb456e.shp')
county.plot()


# <h4><font color='red'> Clipping the Texas County Map to Area of interest </font> </h4>

# In[3]:


AOI1 = ['Newton','Jefferson','Harris','Chambers','Walker','Montgomery',
        'Galveston','Polk','San Jacinto','Liberty','Tyler','Hardin',
        'Jasper','Orange']
AOI = []
for n in AOI1:
    AOI_1 = county.loc[county['name'] == n]
    AOI.append(AOI_1)
    
AOI3 = pd.concat(AOI)
AOI3
AOI3.plot()


# <h4><font color='red'> Clipping the Texas County Map to Jefferson County </font> </h4>

# In[4]:


Jef = ('Jefferson')
Jef = county[county['name'].str.contains(Jef)]
Jef.plot()
Jef.to_file('Jef.tif')


# <h4><font color='red'> Set Working Directory for LULC data 2019 </font> </h4>

# In[5]:


os.chdir("C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/nlcd19_48_lc")
data = gdal.Open('nlcd_2019_land_cover_l48_20210604_TX.img')
ds = rs.open('nlcd_2019_land_cover_l48_20210604_TX.img')


# In[6]:


data.GetProjection()


# <h4><font color='red'> Change the projection of tif file </font> </h4>

# In[7]:


dataReprj = gdal.Warp('lulc_2019.tif', data, dstSRS = 'EPSG:4326')
dataReprj.GetProjection()


# <h4><font color='red'> Clip the Main data to area of interest </font> </h4>

# In[8]:


dataClip = gdal.Warp('jef_2019.tif', data,
                     cutlineDSName = "C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/project2/Texas Counties Map/Jef.tif/Jef.shp",
                     cropToCutline = True, dstNodata = np.nan)


# <h4><font color='red'> Plot the map after clipping </font> </h4>

# In[9]:


arrayClip = dataClip.GetRasterBand(1).ReadAsArray()
plt.imshow(arrayClip)
plt.colorbar()


# <h4><font color='red'> Function to calculate the number of pixel intensities </font> </h4>

# In[10]:


def Count(x,y):
    Count = np.count_nonzero(y == x)
    return(Count) 


# <h4><font color='red'> Function to calculate the area of different variables</font> </h4>

# In[11]:


def Area(Count):
    Area = round((Count*30*30)/2.56e+6,2)   # area in square miles
    return(Area)


# <h4><font color='red'> Total area of Jefferson County </font> </h4>

# In[12]:


Total_count = np.count_nonzero(arrayClip != -128)
Total_area = Area(Total_count)
Total_area


# <h4><font color='red'> Function to calculate the percentage area </font> </h4>

# In[13]:


def Per(Area):
    Percent = (Area/Total_area)*100
    return(round(Percent,2))


# <h4><font color='red'> Calculate the pixel numbers of each class (Jefferson)</font> </h4>

# In[14]:


Water_ice = Count(11,arrayClip) + Count(12,arrayClip)
Developed_land = Count(21,arrayClip) + Count(22,arrayClip)+Count(23,arrayClip) + Count(24,arrayClip)
Barren_land = Count(31,arrayClip)
Forest = Count(41,arrayClip) + Count(42,arrayClip)
Shrubland = Count(51,arrayClip) + Count(52,arrayClip)
Herbaceous = Count(71,arrayClip) + Count(72,arrayClip)+Count(73,arrayClip) + Count(74,arrayClip)
Planted = Count(81,arrayClip) + Count(82,arrayClip)
Vegetation = Shrubland + Herbaceous + Planted
Wetlands = Count(90,arrayClip) + Count(95,arrayClip)
Water = Water_ice + Wetlands


# In[63]:


label = ['Water','Developed_land','Barren_land','Vegetation','Forest']
list = [Water,Developed_land,Barren_land,Vegetation,Forest]
group = np.array(list)
group.tolist()


# <h4><font color='red'> Area of each class for Jefferson county </font> </h4>

# In[65]:


Area_jef = []
for n in group:
    A = Area(n)
    Area_jef.append(A)
Area_jef


# <h4><font color='red'> Percent Area of each class for Jefferson county </font> </h4>

# In[18]:


Per_jef = []
for n in Area_jef:
    P = Per(n)
    Per_jef.append(P)
Per_jef


# <h4><font color='red'> Bar plot showing area of each class for Jefferson county </font> </h4>

# In[73]:


plt.bar(label,Area_jef)
plt.xlabel('Classification')
plt.ylabel('Area(sq miles)')
plt.title('Graphical representation of land use area of Jefferson County')
plt.show() 


# <h4><font color='red'> Pie chart for percent area of Jefferson county </font> </h4>

# In[71]:


plt.pie(Per_jef, autopct = '%1.1f%%')
plt.legend(label)
plt.title('Percent area of land use of jefferson for 2019')
plt.show()


# <h4><font color='red'> Clip and plot the LULC data of 2019 to area of interest </font> </h4>

# In[21]:


dataClip_County = gdal.Warp('lulc19_county.tif', data,
                     cutlineDSName = "C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/project2/Texas Counties Map/County.shp",
                     cropToCutline = True, dstNodata = np.nan)
array_2019 = dataClip_County.GetRasterBand(1).ReadAsArray()
plt.imshow(array_2019)
plt.colorbar()


# <h4><font color='red'> Calculate the pixel numbers of each class (SETX) </font> </h4>

# In[22]:


Water_ice_19 = Count(11,array_2019) + Count(12,array_2019)
Dev_19 = Count(21,array_2019) + Count(22,array_2019)+Count(23,array_2019) + Count(24,array_2019)
Barr_19 = Count(31,array_2019)
Forest_19 = Count(41,array_2019) + Count(42,array_2019)
Shrubland_19 = Count(51,array_2019) + Count(52,array_2019)
Herbaceous_19 = Count(71,array_2019) + Count(72,array_2019)+Count(73,array_2019) + Count(74,array_2019)
Planted_19 = Count(81,array_2019) + Count(82,array_2019)
Veg_19 = Shrubland + Herbaceous + Planted
Wetlands_19 = Count(90,array_2019) + Count(95,array_2019)
Water_19 = Water_ice + Wetlands


# In[23]:


list_19 = [Water_19,Dev_19,Barr_19,Veg_19,Forest_19]
group_19 = np.array(list_19)
G_19 = group_19.tolist()
G_19


# <h4><font color='red'> Area of each class for SETX (2019) </font> </h4>

# In[24]:


Area_19 = []
for n in G_19:
    A_19 = Area(n)
    Area_19.append(A_19)
Area_19   


# <h4><font color='red'> Percent area of each class for SETX(2019) </font> </h4>

# In[25]:


Per_19 = []
for n in Area_19:
    P_19 = Per(n)
    Per_19.append(P_19)
Per_19


# <h4><font color='red'> Set Working Directory for LULC data 2016 </font> </h4>

# In[26]:


os.chdir("C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/NLCD_tRR4jJfr82ZwEnnwnotJ")
data_16 = gdal.Open('NLCD_2016_Land_Cover_L48_20210604_tRR4jJfr82ZwEnnwnotJ.tiff')
data_16.GetProjection()


# <h4><font color='red'> Change the projection of tif file </font> </h4>

# In[27]:


data_16Reprj = gdal.Warp('lulc_16.tif', data_16, dstSRS = 'EPSG:4326')
data_16Reprj.GetProjection()


# <h4><font color='red'> Clip and plot the LULC data of 2016 to area of interest </font> </h4>

# In[28]:


data_County_16 = gdal.Warp('lulc16_county.tif', data_16,
                     cutlineDSName = "C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/project2/Texas Counties Map/County.shp",
                     cropToCutline = True, dstNodata = np.nan)
array_2016 = data_County_16.GetRasterBand(1).ReadAsArray()
plt.imshow(array_2016)
plt.colorbar()


# <h4><font color='red'> Calculate the pixel numbers of each class for SETX(2019) </font> </h4>

# In[29]:


Water_ice_16 = Count(11,array_2016) + Count(12,array_2016)
Dev_16 = Count(21,array_2016) + Count(22,array_2016)+Count(23,array_2016) + Count(24,array_2016)
Barr_16 = Count(31,array_2016)
Forest_16 = Count(41,array_2016) + Count(42,array_2016)
Shrubland_16 = Count(51,array_2016) + Count(52,array_2016)
Herbaceous_16 = Count(71,array_2016) + Count(72,array_2016)+Count(73,array_2016) + Count(74,array_2016)
Planted_16 = Count(81,array_2016) + Count(82,array_2016)
Veg_16 = Shrubland + Herbaceous + Planted
Wetlands_16 = Count(90,array_2016) + Count(95,array_2016)
Water_16 = Water_ice + Wetlands


# In[34]:


list_16 = [Water_16,Dev_16,Barr_16,Veg_16,Forest_16]
group_16 = np.array(list_16)
G_16 = group_16.tolist()
G_16


# <h4><font color='red'> Area of each class for SETX(2016) </font> </h4>

# In[35]:


Area_16 = []
for n in G_16:
    A_16 = Area(n)
    Area_16.append(A_16)
Area_16


# In[62]:


data = [Area_16,Area_19]
data


# <h4><font color='red'> Compare the area of each class between 2016 and 2019 with bar plot </font> </h4>

# In[81]:


X= np.arange(5)
fig = plt.figure()
year = ['2016','2019']
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0,data[0],color = 'b',width = 0.25 )
ax.bar(X+0.25,data[1],color = 'r',width = 0.25 )
ax.set_xlabel('Classification')
ax.set_ylabel('Area(sq miles)')
ax.set_title('Comparison of land use variables for SETX')
ax.set_xticks(X + 0.25)
plt.legend(year)
ax.set_xticklabels(['Water','Developed_land','Barren_land','Vegetation','Forest'])
plt.show()


# <h4><font color='red'> Set Working Directory for LULC data 2001 </font> </h4>

# In[40]:


os.chdir("C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/NLCD_tRR4jJfr82ZwEnnwnotJ")
data_1 = gdal.Open('NLCD_2001_Land_Cover_L48_20210604_tRR4jJfr82ZwEnnwnotJ.tiff')
data_1.GetProjection()


# <h4><font color='red'> Change the projection of tif file </font> </h4>

# In[41]:


data_1Reprj = gdal.Warp('lulc_2001.tif', data_1, dstSRS = 'EPSG:4326')
data_1Reprj.GetProjection()


# <h4><font color='red'> Clip and plot the LULC data of 2001 to area of interest </font> </h4>

# In[42]:


data_County_1 = gdal.Warp('lulc01_county.tif', data_1,
                     cutlineDSName = "C:/Users/Reena Shrestha/OneDrive - lamar.edu/Desktop/Python/project2/Texas Counties Map/County.shp",
                     cropToCutline = True, dstNodata = np.nan)
array_2001 = data_County_1.GetRasterBand(1).ReadAsArray()
plt.imshow(array_2001)
plt.colorbar()


# <h4><font color='red'> Calculate the pixel numbers of each class for SETX(2019) </font> </h4>

# In[43]:


Water_ice_1 = Count(11,array_2001) + Count(12,array_2001)
Dev_1 = Count(21,array_2001) + Count(22,array_2001)+Count(23,array_2001) + Count(24,array_2001)
Barr_1 = Count(31,array_2001)
Forest_1 = Count(41,array_2001) + Count(42,array_2001)
Shrubland_1 = Count(51,array_2001) + Count(52,array_2001)
Herbaceous_1 = Count(71,array_2001) + Count(72,array_2001)+Count(73,array_2001) + Count(74,array_2001)
Planted_1 = Count(81,array_2001) + Count(82,array_2001)
Veg_1 = Shrubland + Herbaceous + Planted
Wetlands_1 = Count(90,array_2001) + Count(95,array_2001)
Water_1 = Water_ice + Wetlands


# In[45]:


list_1 = [Water_1,Dev_1,Barr_1,Veg_1,Forest_1]
group_1 = np.array(list_1)
G_1 = group_1.tolist()
G_1


# <h4><font color='red'> Area of each class for SETX(2001) </font> </h4>

# In[78]:


Area_1 = []
for n in G_1:
    A_1 = Area(n)
    Area_1.append(A_1)
Area_1


# <h4><font color='red'> Percent area of each class for SETX(2001) </font> </h4>

# In[48]:


Per_1 = []
for n in Area_1:
    P_1 = Per(n)
    Per_1.append(P_1)
Per_1


# <h4><font color='red'> Pie chart for percent area of SETX(2001) </font> </h4>

# In[74]:


plt.pie(Per_1, autopct = '%1.1f%%')
plt.legend(label)
plt.title('Percent area of land use of SETX for 2001')
plt.show()


# <h4><font color='red'> Compare the area of each class between 2001, 2016 and 2019 with bar plot </font> </h4>

# In[84]:


Data = [Area_1,Area_16,Area_19]
X= np.arange(5)
fig = plt.figure()
year = ['2001','2019']
ax = fig.add_axes([0,0,1,1])
ax.bar(X+0,Data[0],color = 'b', width = 0.25 )
ax.bar(X+0.25,Data[1],color = 'g', width = 0.25 )
ax.bar(X+0.5,Data[2],color = 'r', width = 0.25 )
ax.set_xlabel('Classification')
ax.set_ylabel('Area(sq miles)')
ax.set_title('Comparison of land use variables for SETX')
ax.set_xticks(X + 0.25)
ax.set_xticklabels(['Water','Developed_land','Barren_land','Vegetation','Forest'])
plt.legend(year)
plt.show()


# In[ ]:





# In[ ]:




