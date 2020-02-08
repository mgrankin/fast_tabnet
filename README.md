# TabNet for fastai
> This is an adaptation of TabNet (Attention-based network for tabular data) for fastai (>=2.0) library. The original paper https://arxiv.org/pdf/1908.07442.pdf. The implementation is taken from here https://github.com/dreamquark-ai/tabnet


## Install

`pip install fast_tabnet`

## How to use

`model = TabNetModel(emb_szs, n_cont, out_sz, embed_p=0., y_range=None, 
                     n_d=8, n_a=8,
                     n_steps=3, gamma=1.3, 
                     n_independent=2, n_shared=2, epsilon=1e-15,
                     virtual_batch_size=128, momentum=0.02)`

Parameters `emb_szs, n_cont, out_sz, embed_p, y_range` are the same as for fastai TabularModel.

- n_d : int
    Dimension of the prediction  layer (usually between 4 and 64)
- n_a : int
    Dimension of the attention  layer (usually between 4 and 64)
- n_steps: int
    Number of sucessive steps in the newtork (usually betwenn 3 and 10)
- gamma : float
    Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
- momentum : float
    Float value between 0 and 1 which will be used for momentum in all batch norm
- n_independent : int
    Number of independent GLU layer in each GLU block (default 2)
- n_shared : int
    Number of independent GLU layer in each GLU block (default 2)
- epsilon: float
    Avoid log(0), this should be kept very low


## Example

Below is an example from fastai library, but the model in use is TabNet

```python
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
```

```python
to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="salary", splits=splits)
```

```python
dbch = to.dataloaders()
dbch.valid.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>age_na</th>
      <th>fnlwgt_na</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Local-gov</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>Black</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>45.000000</td>
      <td>556652.007499</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Never-married</td>
      <td>Sales</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>29.000000</td>
      <td>176683.000072</td>
      <td>13.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>29.000000</td>
      <td>194939.999936</td>
      <td>13.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Transport-moving</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>29.000000</td>
      <td>52635.998841</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>State-gov</td>
      <td>Some-college</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>49.000000</td>
      <td>122177.000557</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>12th</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Other-relative</td>
      <td>Other</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>28.000000</td>
      <td>158737.000048</td>
      <td>8.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Machine-op-inspct</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>55.999999</td>
      <td>192868.999992</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Self-emp-not-inc</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>56.999999</td>
      <td>65080.002276</td>
      <td>9.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Local-gov</td>
      <td>Masters</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>50.000000</td>
      <td>145165.999578</td>
      <td>10.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Never-married</td>
      <td>Tech-support</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>35.000000</td>
      <td>186034.999925</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


```python
to_tst = to.new(df_test)
to_tst.process()
to_tst.all_cols.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>age_na</th>
      <th>fnlwgt_na</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10000</th>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.456238</td>
      <td>1.346622</td>
      <td>1.164335</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10001</th>
      <td>5</td>
      <td>12</td>
      <td>3</td>
      <td>15</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-0.930752</td>
      <td>1.259253</td>
      <td>-0.419996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1.040233</td>
      <td>0.152193</td>
      <td>-1.212162</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>5</td>
      <td>12</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.529237</td>
      <td>-0.282632</td>
      <td>-0.419996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.748235</td>
      <td>1.449564</td>
      <td>0.372169</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
emb_szs = get_emb_sz(to); print(emb_szs)
```

    [(10, 6), (17, 8), (8, 5), (16, 8), (7, 5), (6, 4), (2, 2), (2, 2), (3, 3)]


That's the use of the model

```python
model = TabNetModel(emb_szs, len(to.cont_names), 1, n_d=8, n_a=32, n_steps=1); 
```

```python
opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dbch, model, MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
```

```python
learn.lr_find()
```






![png](docs/images/output_18_1.png)


```python
learn.fit_one_cycle(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.161420</td>
      <td>0.163181</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.140478</td>
      <td>0.127033</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.132842</td>
      <td>0.117864</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.126220</td>
      <td>0.115803</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.125338</td>
      <td>0.117127</td>
      <td>0.757500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.123562</td>
      <td>0.119050</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.121530</td>
      <td>0.117025</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.116976</td>
      <td>0.114524</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.113542</td>
      <td>0.114590</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.111071</td>
      <td>0.114707</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>


## Example with Bayesian Optimization

I like to tune hyperparameters for tabular models with Bayesian Optimization. You can optimize directly your metric using this approach if the metric is sensitive enough (in our example it is not and we use validation loss instead). Also, you should create the second validation set, because you will use the first as a training set for Bayesian Optimization. 


You may need to install the optimizer `pip install bayesian-optimization`

```python
from functools import lru_cache
```

```python
# The function we'll optimize
@lru_cache(1000)
def get_accuracy(n_d:Int, n_a:Int, n_steps:Int):
    model = TabNetModel(emb_szs, len(to.cont_names), 1, n_d=n_d, n_a=n_a, n_steps=n_steps);
    learn = Learner(dbch, model, MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
    learn.fit_one_cycle(5)
    return -float(learn.validate(dl=learn.dls.valid)[0])
```

This implementation of Bayesian Optimization doesn't work naturally with descreet values. That's why we use wrapper with `lru_cache`.

```python
def fit_accuracy(pow_n_d, pow_n_a, pow_n_steps):
    pow_n_d, pow_n_a, pow_n_steps = map(int, (pow_n_d, pow_n_a, pow_n_steps))
    return get_accuracy(2**pow_n_d, 2**pow_n_a, 2**pow_n_steps)
```

```python
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'pow_n_d': (0, 8), 'pow_n_a': (0, 8), 'pow_n_steps': (0, 4)}

optimizer = BayesianOptimization(
    f=fit_accuracy,
    pbounds=pbounds,
)
```

```python
optimizer.maximize(
    init_points=15,
    n_iter=100,
)
```

    |   iter    |  target   |  pow_n_a  |  pow_n_d  | pow_n_... |
    -------------------------------------------------------------



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.376397</td>
      <td>0.227991</td>
      <td>0.757500</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.307311</td>
      <td>0.188101</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.192308</td>
      <td>0.174029</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.180625</td>
      <td>0.168215</td>
      <td>0.757500</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.171093</td>
      <td>0.168311</td>
      <td>0.757500</td>
      <td>00:07</td>
    </tr>
  </tbody>
</table>






    | [0m 1       [0m | [0m-0.1683  [0m | [0m 2.099   [0m | [0m 2.108   [0m | [0m 2.532   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.156191</td>
      <td>0.145624</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.135885</td>
      <td>0.131468</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.124489</td>
      <td>0.116068</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.120778</td>
      <td>0.115556</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.118399</td>
      <td>0.114798</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [95m 2       [0m | [95m-0.1148  [0m | [95m 5.582   [0m | [95m 0.5914  [0m | [95m 0.394   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.732101</td>
      <td>0.201414</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.213341</td>
      <td>0.182902</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.157790</td>
      <td>0.154676</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.143525</td>
      <td>0.134003</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.137171</td>
      <td>0.128810</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>






    | [0m 3       [0m | [0m-0.1288  [0m | [0m 0.6418  [0m | [0m 3.424   [0m | [0m 3.649   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.255437</td>
      <td>0.176615</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.164086</td>
      <td>0.158516</td>
      <td>0.757500</td>
      <td>00:07</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.149184</td>
      <td>0.139764</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.137243</td>
      <td>0.126479</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.132500</td>
      <td>0.125504</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>






    | [0m 4       [0m | [0m-0.1255  [0m | [0m 6.121   [0m | [0m 1.372   [0m | [0m 2.897   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.834591</td>
      <td>0.252279</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.233243</td>
      <td>0.190753</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.174514</td>
      <td>0.163240</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.160865</td>
      <td>0.149085</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.153380</td>
      <td>0.142670</td>
      <td>0.757500</td>
      <td>00:06</td>
    </tr>
  </tbody>
</table>






    | [0m 5       [0m | [0m-0.1427  [0m | [0m 7.183   [0m | [0m 5.46    [0m | [0m 2.131   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.280760</td>
      <td>0.184326</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.151150</td>
      <td>0.149422</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.136892</td>
      <td>0.126405</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.129048</td>
      <td>0.124096</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.129486</td>
      <td>0.122428</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [0m 6       [0m | [0m-0.1224  [0m | [0m 0.5754  [0m | [0m 2.298   [0m | [0m 1.447   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2.923816</td>
      <td>0.290585</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.635441</td>
      <td>0.237105</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.272063</td>
      <td>0.170947</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.179265</td>
      <td>0.156215</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.159060</td>
      <td>0.151041</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>






    | [0m 7       [0m | [0m-0.151   [0m | [0m 6.365   [0m | [0m 7.881   [0m | [0m 3.652   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.436597</td>
      <td>0.213113</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.350264</td>
      <td>0.189146</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.187943</td>
      <td>0.162571</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.165730</td>
      <td>0.154995</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.155386</td>
      <td>0.149732</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>






    | [0m 8       [0m | [0m-0.1497  [0m | [0m 5.544   [0m | [0m 5.838   [0m | [0m 3.925   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.430938</td>
      <td>0.227863</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.209979</td>
      <td>0.177186</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.179570</td>
      <td>0.164046</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.170003</td>
      <td>0.161813</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.168120</td>
      <td>0.159528</td>
      <td>0.757500</td>
      <td>00:10</td>
    </tr>
  </tbody>
</table>






    | [0m 9       [0m | [0m-0.1595  [0m | [0m 4.231   [0m | [0m 1.842   [0m | [0m 3.959   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.196750</td>
      <td>0.168031</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.155173</td>
      <td>0.152989</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.144540</td>
      <td>0.126592</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.133649</td>
      <td>0.126462</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.124242</td>
      <td>0.119457</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [0m 10      [0m | [0m-0.1195  [0m | [0m 7.513   [0m | [0m 6.718   [0m | [0m 0.3416  [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.244845</td>
      <td>0.162216</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.152529</td>
      <td>0.143890</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.134093</td>
      <td>0.119078</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.123151</td>
      <td>0.114896</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.118943</td>
      <td>0.114241</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [95m 11      [0m | [95m-0.1142  [0m | [95m 0.1011  [0m | [95m 7.301   [0m | [95m 0.3318  [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.142999</td>
      <td>0.161638</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.133412</td>
      <td>0.120533</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.126682</td>
      <td>0.119148</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.124139</td>
      <td>0.114975</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.120500</td>
      <td>0.115238</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [0m 12      [0m | [0m-0.1152  [0m | [0m 1.466   [0m | [0m 1.523   [0m | [0m 0.6053  [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.292738</td>
      <td>0.216998</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.342039</td>
      <td>0.196274</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.207225</td>
      <td>0.155081</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.170700</td>
      <td>0.151608</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.156820</td>
      <td>0.148455</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>






    | [0m 13      [0m | [0m-0.1485  [0m | [0m 3.104   [0m | [0m 6.879   [0m | [0m 3.043   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.248299</td>
      <td>0.242661</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.248616</td>
      <td>0.242500</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.247367</td>
      <td>0.242500</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.243612</td>
      <td>0.242500</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.248120</td>
      <td>0.242500</td>
      <td>0.757500</td>
      <td>00:05</td>
    </tr>
  </tbody>
</table>






    | [0m 14      [0m | [0m-0.2425  [0m | [0m 6.56    [0m | [0m 0.3284  [0m | [0m 1.299   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.421470</td>
      <td>0.166623</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.186513</td>
      <td>0.165642</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.152764</td>
      <td>0.137757</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.142408</td>
      <td>0.135240</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.138158</td>
      <td>0.131554</td>
      <td>0.757500</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>






    | [0m 15      [0m | [0m-0.1316  [0m | [0m 0.632   [0m | [0m 4.901   [0m | [0m 3.857   [0m |



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.167871</td>
      <td>0.171308</td>
      <td>0.757500</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.142500</td>
      <td>0.133382</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.127953</td>
      <td>0.120554</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.122714</td>
      <td>0.118568</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.123305</td>
      <td>0.116220</td>
      <td>0.757500</td>
      <td>00:04</td>
    </tr>
  </tbody>
</table>






    | [0m 16      [0m | [0m-0.1162  [0m | [0m 4.026   [0m | [0m 4.246   [0m | [0m 0.0     [0m |




    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='2' class='' max='5', style='width:300px; height:20px; vertical-align: middle;'></progress>
      40.00% [2/5 00:32<00:48]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>5.261782</td>
      <td>0.234573</td>
      <td>0.757500</td>
      <td>00:16</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.748920</td>
      <td>0.271949</td>
      <td>0.757500</td>
      <td>00:16</td>
    </tr>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='102' class='' max='125', style='width:300px; height:20px; vertical-align: middle;'></progress>
      81.60% [102/125 00:12<00:02 0.3221]
    </div>



```python
optimizer.max
```
