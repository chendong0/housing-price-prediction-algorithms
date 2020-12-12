#!/usr/bin/env python
# coding: utf-8

# ## 2.基于多个回归预测算法的房价预测 
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import math
from scipy import stats
from scipy.stats import norm
import seaborn
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#002
df_train = pd.read_csv('./data/data_train.csv',encoding='gbk')
'''
方法的命名则遵循函数 命名规则，都是小写，并用下划线分隔。
用read读取文件路径
encoding'gbk'编码方式为中文
'''
df_train #显示数据集(应该是省略了print()前缀)


# In[3]:


#003
print(df_train.columns) #打印出列索引,打印出所有属性,'房价'是标签,不是属性


# In[4]:


#004
print(df_train['房价'].describe()) #经典的describe,对房价进行描述性统计分析
print("\nSkewness: %f" % df_train['房价'].skew()) 

#\n表示回车换行
#df_train['房价']把所有数据集中的房价拿出来
#skew计算偏度值,数据偏右
print("Kurtosis: %f" % df_train['房价'].kurt()) #计算峰度值,峰部为尖峰.
myfont=FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf') #设置中文字体,simhei是黑体
seaborn.set(font=myfont.get_name()) # 设置绘图风格
#seaborn是matplotlib.pyplot的升级版.

plt.figure(figsize=(16, 8),dpi=600) #设置图片大小为了美观,dpi为分辨率
plt.ylabel('频数') #设置纵轴标签,y轴显示的文字
plt.title('房价分布') #设置图片标题,x轴显示的文字

# displot函数,给定一组连续值的数据,将他们分成若干小段,统计每个小段中数据的个数,并画出它们的直方图
#若kde=True则绘制拟合曲线,(核密度估计曲线),hist_kws设置柱状图的图例标签为频数
seaborn.distplot(df_train['房价'],kde=False,hist_kws={"label":"频数"})
#kdeplot核密度估计(kernel density estimation)是用在概率论中用来估计未知的密度函数,属于
#非参数检验方法之一.
plt.legend() #显示频数


# In[5]:


data = pd.concat([df_train['房价'], df_train['居住面积']], axis=1)
#用数据把'房价'和'居住面积'拿出来,拼接两个序列
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小,分辨率
seaborn.scatterplot(data['居住面积'],data['房价']) #做散点图


# In[6]:


data = pd.concat([df_train['房价'], df_train['地下室总面积']], axis=1)
#拼接两个序列
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小和分辨率
seaborn.scatterplot(data['地下室总面积'],data['房价']) #作散点图


# In[7]:


data = pd.concat([df_train['房价'], df_train['材料和质量']], axis=1)
#拼接两个序列
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小和分辨率
seaborn.boxplot(data=data,x='材料和质量', y="房价") #画出盒图


# In[8]:


data = pd.concat([df_train['房价'], df_train['原施工日期']], axis=1)
#拼接两个序列
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小和分辨率
plt.xticks(rotation=90) #将横坐标旋转90度
seaborn.boxplot(data=data,x='原施工日期', y="房价") #画出盒图


# In[9]:


data = pd.concat([df_train['房价'], df_train['街区']], axis=1)
#拼接两个序列图 
#axis = 1 代表对纵轴操作,也就是第一轴
plt.figure(figsize=(16, 8),dpi=600)#设置图片大小和分辨率
plt.xticks(rotation=90) #横坐标旋转90度
seaborn.boxplot(data=data,x='街区', y="房价" ) #画出盒图


# In[10]:


corrmat = df_train.corr() #计算相关系数矩阵,cooperation relation
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小和分辨率
plt.rcParams['axes.unicode_minus'] = False #解决图像中的符号'-'显示为方框的问题
#corrmat为相关系数矩阵,square=True代表热力图中的格子为正方形,camp='YlGnBu'代表热力图颜色有黄绿蓝构成
#xticklabels代表x轴的刻度上显示的值
seaborn.heatmap(corrmat,square=True,cmap='YlGnBu',xticklabels=True,yticklabels=True)
#画出热力图,square=True,正方形等于真,就是画出正方形,cmap黄色绿色蓝色,xlabel和ylabel两条轴线画出来
#对角线颜色深,是自己对自己的颜色肯定是100%


# In[11]:


k = 10 #热力图中变量个数,因图太密了,只找出10个最大的相关性变量
#nlargest代表first n rows with the largest values
cols = corrmat.nlargest(k, '房价')['房价'].index
#corrmat用斜方差矩阵,再用nlargest
#取出与房价的相关系数排名靠前的10个特征的名称
#corrcoef中rowvar参数默认为True,代表把每行当作一个变量,因此传入的列向量构成的矩阵需要转置
cm = np.corrcoef(df_train[cols].values.T)
#corr和corrcoef都是计算相关系数矩阵的函数,前者作用于dataframe的数据结构,后者作用于数组
plt.figure(figsize=(16, 8),dpi=600) #设置图片大小
#画出热力图,cm为相关系数矩阵,cbar指是否使用颜色条做为图例,annot指是否在热力图的每个单元上显示数值.
#square指图中格子是否显示成方形,fmt指数字保留两位小数,annot_kws指定数字大小
hm = seaborn.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, 
                 yticklabels=cols.values, xticklabels=cols.values,cmap='YlGnBu')


# In[12]:


total_isnull = df_train.isnull().sum().sort_values(ascending=False)
#检查df_train数据表格中是否isnull有空,只要表格没有数据就表示为False,有数据为True
#统计每一列有多少个isnull空值,统计完按values空值个数排序
#统计各属性的缺失个数,并按照降序排序
total_isnull#print()


# In[13]:


total_number_rows=len(df_train) #计算总行数
percent = (total_isnull/total_number_rows)*100
#用每一列空的总数除以总行数
#计算各属性缺失值所占的比例

missing_data = pd.concat([total_isnull, percent], axis=1, keys=['缺失值数量', '百分比'])
#concat连接的方法连接起来,拼接序列
missing_data.head(20)
#显示前20行


# In[14]:


train = pd.read_csv('./data/data_train.csv',encoding='gbk')
test = pd.read_csv('./data/data_test.csv',encoding='gbk')

print("The train data size before dropping Id feature is : （%d %d）" %(train.shape[0],test.shape[1]))
print("The test data size before dropping Id feature is : （%d %d）\n" %(test.shape[0],test.shape[1]))

train.drop("编号", axis = 1, inplace = True)
test.drop("编号", axis = 1, inplace = True)

print("The train data size after dropping Id feature is : （%d %d）" %(train.shape[0],test.shape[1]))
print("The train data size after dropping Id feature is : （%d %d）" %(test.shape[0],test.shape[1]))


# In[15]:


plt.figure(figsize=(16, 8),dpi=600)
fig = seaborn.scatterplot(train['居住面积'],train['房价'])


# In[16]:


train = train.drop(train[(train['居住面积']>4000) & (train['房价']<300000)].index)

plt.figure(figsize=(16, 8),dpi=600)
fig = seaborn.scatterplot(train['居住面积'],train['房价'])


# In[17]:


(mu, sigma) = norm.fit(train['房价'])
print("\n mu = %.2f and sigma = %.2f\n" %(mu,sigma))

plt.figure(figsize=(16, 8),dpi=600)
plt.ylabel('概率密度')
plt.title('房价分布')
seaborn.distplot(train['房价'] ,fit=norm,kde=True,fit_kws={"label":"正态分布概率密度曲线"},
                 kde_kws={"label":"核密度估计曲线"},hist_kws={"label":"房价频数"})
plt.legend()

plt.figure(figsize=(16, 8),dpi=600)#
res = stats.probplot(train['房价'], plot=plt)


# In[18]:


train["房价"] = np.log1p(train["房价"])

(mu, sigma) = norm.fit(train['房价'])
print("\n mu = %.2f and sigma = %.2f\n" %(mu,sigma))

plt.figure(figsize=(16, 8),dpi=600)
plt.ylabel('概率密度')
plt.title('房价分布')
seaborn.distplot(train['房价'] ,fit=norm,kde=True,fit_kws={"label":"正态分布概率密度曲线"},
                 kde_kws={"label":"核密度估计曲线"},hist_kws={"label":"房价频数直方图"})
plt.legend()


plt.figure(figsize=(16, 8),dpi=600)
res = stats.probplot(train['房价'], plot=plt)


# In[19]:


ntrain = train.shape[0]
y_train = train.房价.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['房价'], axis=1, inplace=True)
print("all_data size is :(%d,%d) " %(all_data.shape[0],all_data.shape[1]))


# In[20]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na


# In[21]:


all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:10]
missing_data = pd.DataFrame({'缺失值占比' :all_data_na})
missing_data


# In[22]:


plt.figure(figsize=(16, 8),dpi=600)
plt.xticks(rotation='90')
plt.xlabel('特征')
plt.ylabel('缺失值占比')
plt.title('各个特征的缺失值占比图')
seaborn.barplot(x=all_data_na.index, y=all_data_na)


# In[23]:


cols = ('泳池质量','车库类型', '车库室内装修', '车库质量', '车库条件','地下室的高度','地下室的条件','花园层地下室墙',
        '地下室基底质量','地下室第二成品区质量','砌体饰面型','砌体饰面型','杂项功能','通道入口的类型',
        '栅栏的质量','壁炉质量','建筑类','实用工具')                      

for col in cols:
    all_data[col] = all_data[col].fillna('None')
    
all_data['泳池质量']


# In[24]:


list(all_data.groupby('街区')['街道连接距离'])[0:3]


# In[25]:


all_data['街道连接距离'] = all_data.groupby('街区')['街道连接距离'].transform(lambda x: x.fillna(x.median()))
all_data['街道连接距离']


# In[26]:


cols = ('车库修筑年份','车库规模','车库车位数量','地下室基底面积','地下室第二成品区面积',
        '未使用的地下室面积','地下室总面积','地下室全浴室','地下室半浴室','砌体饰面面积','砌体饰面面积')                     
for col in cols:
    all_data[col] = all_data[col].fillna(0)
    
all_data['车库规模']


# In[27]:


cols = ('一般分区','电气系统','厨房的品质','第一外部材料','第二外部材料','家庭功能评级','销售类型')                    
for col in cols:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0]) 
all_data['厨房的品质']


# In[28]:


all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({' 缺失值占比' :all_data_na})
missing_data.head()


# In[29]:


c='壁炉质量'
all_data_c_list=list(all_data[c].values)
all_data_c_list_df=pd.DataFrame(all_data_c_list,columns=['beforeTransform'])

from sklearn.preprocessing import LabelEncoder                 
myLabel = LabelEncoder()                                 
myLabel.fit(all_data_c_list)                            
all_data_c_list_transform= myLabel.transform(all_data_c_list)
all_data_c_list_transform_df=pd.DataFrame(all_data_c_list_transform,columns=['afterTransform'])

pd.concat([all_data_c_list_df,all_data_c_list_transform_df],axis=1).head(15)


# In[30]:


print(all_data['建筑类'])

all_data['建筑类'] = all_data['建筑类'].apply(str)
all_data['总体状况评价'] = all_data['总体状况评价'].apply(str)
all_data['销售年'] = all_data['销售年'].apply(str)
all_data['销售月'] = all_data['销售月'].apply(str)


# In[32]:


from sklearn.preprocessing import LabelEncoder
cols = ('壁炉质量', '地下室的高度', '地下室的条件', '车库质量', '车库条件', 
        '外部材料质量', '外部材料条件','加热质量', '泳池质量', '厨房的品质', '地下室基底质量', 
        '地下室第二成品区质量', '家庭功能评级', '栅栏的质量', '花园层地下室墙', '车库室内装修', '坡的财产',
        '财产的类型', '车道', '道路类型', '通道入口的类型', '中央空调', '建筑类', '总体状况评价', 
        '销售年', '销售月')                      

for c in cols:
    myLabel = LabelEncoder()
    myLabel.fit(list(all_data[c].values))
    all_data[c] = myLabel.transform(list(all_data[c].values))
        
print('Shape all_data: (%d,%d)'%(all_data.shape[0],all_data.shape[1]))    


# In[33]:


all_data['总面积'] = all_data['地下室总面积'] + all_data['一楼面积'] + all_data['二楼面积']


# In[34]:


all_data.dtypes


# In[35]:


from scipy.stats import norm, skew
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
print("\n所有数值型特征的偏度为: \n")
skewness = pd.DataFrame({'偏度值' :skewed_feats})
skewness.head(10)


# In[36]:


skewness = skewness[abs(skewness) > 0.75]
print("总共有 %d 个特征需要进行BOX-COX变换" %skewness.shape[0])
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15 
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)


# In[37]:


sale_contion=all_data['销售条件']
sale_contion


# In[38]:


sale_contion_afterDummies=pd.get_dummies(sale_contion)
sale_contion_afterDummies


# In[39]:


all_data = pd.get_dummies(all_data)
print(all_data.shape)
all_data


# In[40]:


train = all_data[:ntrain]
test = all_data[ntrain:]


# In[41]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score,train_test_split


# In[42]:


Lasso = Lasso(alpha =0.0005, random_state=0)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2)

ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=0)

GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt', random_state =0)

XGBR = xgb.XGBRegressor(n_estimators=2200,learning_rate=0.05, max_depth=3, 
                        reg_alpha=0.4640, reg_lambda=0.8571)


# In[43]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        for model in self.models_:
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[52]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred     
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[53]:


n_folds = 5
def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[54]:


score = rmse_cv(Lasso)
print("\nLasso score: %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[55]:


score = rmse_cv(ENet)
print("ElasticNet score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[56]:


score = rmse_cv(KRR)
print("Kernel Ridge score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[57]:


score = rmse_cv(GBR)
print("Gradient Boosting score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[58]:


score = rmse_cv(XGBR)
print("Xgboost score: %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[59]:


averaged_models = AveragingModels(models = (ENet, GBR, KRR, Lasso))

score = rmse_cv(averaged_models)
print(" Averaged base models score: %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[60]:


stacked_averaged_models1 = StackingAveragedModels(base_models = (ENet, GBR, KRR),
                                                 meta_model = Lasso)

score = rmse_cv(stacked_averaged_models1)
print("Stacking Averaged models1 score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[61]:


stacked_averaged_models2 = StackingAveragedModels(base_models = (Lasso, GBR, KRR),
                                                 meta_model = ENet)
 
score = rmse_cv(stacked_averaged_models2)
print("Stacking Averaged models2 score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[62]:


stacked_averaged_models3 = StackingAveragedModels(base_models = (ENet, Lasso, KRR),
                                                 meta_model = GBR)

score = rmse_cv(stacked_averaged_models3)#计算均方根误差
print("Stacking Averaged models3 score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# In[63]:


stacked_averaged_models4 = StackingAveragedModels(base_models = (ENet, GBR, Lasso),
                                                 meta_model = KRR)

score = rmse_cv(stacked_averaged_models4)
print("Stacking Averaged models4 score:  %.4f (%.4f)\n" %(score.mean(),score.std()))


# **5.模型应用**

# In[64]:


stacked_averaged_models4.fit(np.array(train),y_train)
y_pred=stacked_averaged_models4.predict(test)


# In[65]:


y_pred_real=[]
for i in range(len(y_pred)):
    y_pred_real_i=pow(math.exp(1),y_pred[i])-1
    y_pred_real.append(y_pred_real_i)


# In[66]:


pd.DataFrame(y_pred_real).describe()


# In[67]:


plt.figure(figsize=(20,8),dpi=600)
plt.plot(y_pred_real[0:200],'bo')
plt.xlabel('样本编号')
plt.ylabel('预测房价')
plt.show()


# In[ ]:




