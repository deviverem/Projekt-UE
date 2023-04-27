**Temat**: Dobrostan psychiczny i uwarunkowania demograficzno-społeczne a poczucie szczęścia mieszkańców różnych państw - eksploracyjna analiza danych.

#Cel projektu

#Przygotowanie środowiska

Importuje biblioteki, z których będę korzystała podczas analizy danych.

import pandas as pd    
import numpy as np
from scipy import stats
import scipy.stats 
from scipy.stats import shapiro 
!pip install pingouin --upgrade
import pingouin as pg
from pingouin import welch_anova
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns

#import warnings
#warnings.filterwarnings('ignore')

"""#Import i przygotowanie danych

Przystępuję do wczytania danych, wyboru potrzebnych zmiennych, łączenia ramek danych, uzupełniania braków w danych.
"""

from google.colab import files
uploaded=files.upload()

data=pd.read_csv('Data2022.csv', delimiter=',')
df_happiness=data.copy().sort_values(['Country name', 'year'])
df_happiness

df_happiness.info()

gdppercapita=pd.read_csv('gdp_per_capita.csv', delimiter=',')
df_gdpwide=gdppercapita.copy()
df_gdpwide.head(10)

df_gdpwide.columns    # przygotowanie do zmiany formatu ramki danych z szerokiego na długi

df_gdp=df_gdpwide.melt(id_vars=['Country Name', 'Code'], value_vars=['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001','2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010','2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019','2020'], var_name=['year'], value_name='gdp per capita')
df_gdp=df_gdp.sort_values(['Country Name', 'year'])
df_gdp

df_gdp.info()

df_gdp['year']=df_gdp['year'].apply(pd.to_numeric)  # zmieniam typ danych na INT, aby możliwe było połączenie ramek danych poprzez wskazaną kolumnę

df_happiness=pd.merge(df_happiness, df_gdp, how='left', left_on=['Country name', 'year'], right_on=['Country Name', 'year'])
df_happiness

"""W kolejnych krokach usunę niepotrzebne kolumny, zmienię nazwy kolumn (m.in. ujednolicę format ich zapisu) oraz sprawdzę, czy typy danych są właściwe do działań, jakie będą na nich wykonywane, a także w jakich kolumnach występują braki danych."""

df_happiness.info()

df_happiness.drop(['Country Name', 'Log GDP per capita'], axis=1, inplace=True)

df_happiness.columns=map(str.lower, df_happiness.columns)

df_happiness.rename({'country name': 'country','code':'alpha-3'}, axis=1, inplace=True) 
df_happiness=df_happiness.sort_values(['country', 'year'])

df_happiness.info()

df_happiness['year'].min(), df_happiness['year'].max(), len(df_happiness['country'].unique())

Kolumny zawierające braki (oprócz tej, która zawiera dane o GDP per capita) przechowują informacje na temat tego jaka YYYYYYYYYY część wskaźnika satysfakcji jest wyjaśniana przez poszczgólną zmienną. Obserwowane braki danych nie są istotne dla dalszej analizy, ponieważ...........
"""

mentsubs=pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv', delimiter=',')
df_ment_subs=mentsubs.copy().sort_values(['Entity', 'Year'])
df_ment_subs

df_ment_subs.rename({'Entity':'country',
                     'Code':'alpha-3',
                     'Year':'year', 
                     'Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)':'schizophrenia',
                     'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)':'bipolar disorder',
                     'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)':'eating disorders',
                     'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)':'anxiety disorders',
                     'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)':'drug use disorders',
                     'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)':'depressive disorders',
                     'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)':'alcohol use disorders'}, 
                    axis=1, inplace=True)
df_ment_subs.head()

df_ment_subs.info()   # późniejsze łączenie kilku ramek danych może rozwiązać kwestię brakujących wartości w kolumnie 'alpha-3'

df_ment_subs['year'].min(), df_ment_subs['year'].max(), len(df_ment_subs['country'].unique())

suic=pd.read_csv('suicide-death-rates-by-sexY.csv', delimiter=',')
df_suicide=suic.copy()  
df_suicide

df_suicide.info()

df_suicide['Year'].min(), df_suicide['Year'].max(), len(df_suicide['Entity'].unique())

depr = pd.read_csv('prevalence-of-depression-males-vs-females.csv', delimiter=',')
df_depression=depr.copy()
df_depression

df_depression.info()    # zastanawia mnie znacząca liczba wierszy tej ramki oraz spora róźnica w liczbie danych jakie zawierają poszczególne kolumny: szukam przyczyn

df_depression['Year'].min(), df_depression['Year'].max(), len(df_depression['Entity'].unique())   # liczba przekracza wszystkie możliwe listy państw świata, sprawdzam co się w niej kryje

list(df_depression['Entity'].unique())

"""Liczba rekordów w badanej DataFrame jest bliska 58 tysięcy, ponieważ zawiera dane pochodzące z lat 10000 p.n.e - 2021 n.e. oraz 321 krajów (np. nieistniejących, jak USSR). Nadmiarowe dane w szybki sposób zostaną wyłączone z dalszej analizy poprzez łączenie tej ramki danych z danymi o samobójstwach. """

df_depr_suic=pd.merge(df_suicide, df_depression, how='inner', on=['Entity', 'Year'])
df_depr_suic.head()

df_depr_suic.info()     # podobnie jak poprzednio, brakującymi wartościami zajmę się po połączeniu ramek danych

df_depr_suic[df_depr_suic['Code_x'].isnull()].equals(df_depr_suic[df_depr_suic['Code_y'].isnull()])      # porównywane kolumny zawierają braki danych w tych samych wierszach, więc usunę dowolną z nich (oraz niepotrzebną zmienną 'Continent')

df_depr_suic.drop(['Code_y', 'Continent'], axis=1, inplace=True)

df_depr_suic.rename({'Entity':'country', 
                   'Year':'year',
                   'Code_x':'alpha-3',                    
                   'Male suicide rate (age-standardized)':'male suicide rate',
                   'Female suicide rate (age-standardized)':'female suicide rate',
                   'Deaths - Self-harm - Sex: Both - Age: Age-standardized (Rate)':'deaths by self-harm rate (both sexes)',
                   'Prevalence - Depressive disorders - Sex: Male - Age: Age-standardized (Percent)':'male depressive disorder (percent)',
                   'Prevalence - Depressive disorders - Sex: Female - Age: Age-standardized (Percent)':'female depressive disorder (percent)',
                   'Population (historical estimates)':'population'}, axis=1, inplace=True)
df_depr_suic.head()

df_depr_suic=df_depr_suic.astype({'population': 'Int64'})

df_depr_suic['year'].min(), df_depr_suic['year'].max(), len(df_depr_suic['country'].unique())

mentals=pd.read_csv('suicide-rates-vs-prevalence-of-mental-and-substance-use-disorders.csv', delimiter=',')                                 
df_suic_mental=mentals.copy()
df_suic_mental

df_psychopathology=pd.merge(df_ment_subs, df_depr_suic, how='inner', on=['country', 'year', 'alpha-3'])
df_psychopathology

df_psychopathology.info()

df_psychopathology['year'].min(), df_psychopathology['year'].max(), len(df_psychopathology['country'].unique())

"""W kolejnym kroku przyjrzę się rekordom, które zawierają w sobie braki danych: najpierw dla kolumn opisujących wskaźnik samobójstw dla kobiet i mężczyzn, potem te, które występują w danych dotyczących liczebności populacji badanych krajów. Wartościom NaN w kolumnmie 'alpha-3' przyjrzę się na końcu."""

f_suic_null=df_psychopathology[df_psychopathology['female suicide rate'].isnull()]
f_suic_null

m_suic_null=df_psychopathology[df_psychopathology['male suicide rate'].isnull()]
m_suic_null['male suicide rate'].equals(f_suic_null['female suicide rate'])

suic_bef_2018=f_suic_null.loc[f_suic_null['year']<2018]                   
suic_bef_2018

suic_aft_2018=f_suic_null.loc[f_suic_null['year']>=2018]
suic_aft_2018

"""Analiza pierwszej z trzech powyższych ramek danych prowadzi do kilku wniosków, które potwierdziłam w kolejnych linijkach kodu, a mianowicie, że brakujące wartości w kolumnach wskaźnika samobójstw kobiet i mężczyzn:
*   pojawiają się w tych samych rekordach,
*   pojawiają się na przestrzeni wszystkich raportowanych lat m.in. w rejonach świata, tj. 'Africa Region (WHO)' (całościowy brak danych wyklucza możliwość zostawienia tych wierszy w mojej ramce danych),
*   występują także w szeregu innych krajów jedynie dla lat 2018 i 2019 (uzupełnię je danymi skopiowanymi z roku poprzedzającego wypełniany).
"""

index_to_drop=df_psychopathology[(df_psychopathology['male suicide rate'].isnull())&(df_psychopathology['year']<2018)].index
df_psychopathology.drop(index_to_drop, inplace=True)

countries_to_drop=set(list(suic_bef_2018['country'])).intersection(list(suic_aft_2018['country']))       #   jest Wales i England
index_to_drop2=df_psychopathology[((df_psychopathology['female suicide rate'].isnull())|(df_psychopathology['year']>=2018))&(df_psychopathology['country'].isin(countries_to_drop))].index      
df_psychopathology.drop(index_to_drop2, inplace=True)

df_psychopathology.info()

to_fill=['female suicide rate', 'male suicide rate']
df_psychopathology[to_fill]=df_psychopathology[to_fill].ffill()
df_psychopathology.info()

df_psychopathology['year'].min(), df_psychopathology['year'].max(), len(df_psychopathology['country'].unique())

"""Usunęłam z ramki danych wszystkie rekordy nie zawierające danych o wskaźniku samobójstw dla kobiet i mężczyzn, ponieważ byłyby one zbędne w dalszych etapach analizy. W wyniku wykonania tej operacji w DataFrame zniknęły także puste wartości z kolumn *alpha-3* i *population* (co oznacza, że znajdowały się one wyłącznie w usuniętych wierszach). Ramka danych dotycząca psychopatologii zawierają teraz 5432 rekordy, dla 194 krajów, opisujące lata 1990-2017."""

df_all=pd.merge(df_happiness[['country', 'year', 'life ladder', 'gdp per capita','healthy life expectancy at birth']], df_psychopathology, how='inner', on=['country', 'year'],)
df_all

df_all.info()

"""Zbiorcza ramka danych *df_all* zawiera 1768 wierszy z danymi zabranymi w 19 kolumnach. Typy danych odpowiadają zawartości kolumn, nie wymagają zmian. W 136 rekordach brakuje danych o produkcie krajowym brutto na mieszkańca, uzupełnię je danymi ze strony internetowej Banku Światowego."""

df_all[df_all['gdp per capita'].isnull()]

nowe=pd.read_csv('5ccc4db6-7673-414d-8d2a-bb75d91f1dad_Data.csv', delimiter=',')
df_nowe=nowe.copy()
df_nowe.head()

df_nowe.drop(['Series Name', 'Series Code'], axis=1, inplace=True)
df_nowe.drop(index=[14, 15, 16, 17, 18], axis=0, inplace=True)

for col in df_nowe.columns[2:]:
    df_nowe.rename(columns={col:col.split(' ')[0]}, inplace=True)
df_nowe

df_nowe_melted=df_nowe.melt(id_vars=['Country Name', 'Country Code'], value_vars=df_nowe.columns[2:], var_name=['year'], value_name='gdp per capita')
df_nowe_melted=df_nowe_melted.sort_values(['Country Name', 'year'])
df_nowe_melted

df_nowe_melted['gdp per capita']=df_nowe_melted['gdp per capita'].replace('..', np.nan)
df_nowe_melted.info()

df_nowe_melted[df_nowe_melted['gdp per capita'].isnull()]

df_nowe_melted[['year', 'gdp per capita']]=df_nowe_melted[['year', 'gdp per capita']].apply(pd.to_numeric)  # zmieniam typ danych na INT, aby możliwe było połączenie ramek danych poprzez wskazaną kolumnę

df_nowe_melted.rename({'Country Code':'alpha-3'}, axis=1, inplace=True)
df_nowe_melted.info()

df_all=pd.merge(df_all, df_nowe_melted[['year', 'alpha-3', 'gdp per capita']], how='left', on=['year', 'alpha-3'])
df_all

df_all['gdp per capita']=df_all['gdp per capita_x'].fillna(df_all['gdp per capita_y'])
df_all.drop(['gdp per capita_x', 'gdp per capita_y'], axis=1, inplace=True)
df_all.info()

df_all['gdp per capita']=df_all['gdp per capita'].ffill()
df_all.info()

depr_age = pd.read_csv('prevalence-of-depression-by-age.csv', delimiter=',')
df_depr_age=depr_age.copy()
df_depr_age

df_depr_age.info()

df_depr_age.drop(['Prevalence - Depressive disorders - Sex: Both - Age: 20 to 24 (Percent)', 
                  'Prevalence - Depressive disorders - Sex: Both - Age: 30 to 34 (Percent)',
                  'Prevalence - Depressive disorders - Sex: Both - Age: 15 to 19 (Percent)',
                  'Prevalence - Depressive disorders - Sex: Both - Age: 25 to 29 (Percent)',
                  'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)'], axis=1, inplace=True)
df_depr_age.head()

death_age=pd.read_csv('suicide-rates-by-age-detailed----self harm.csv', delimiter=',')
df_death_age=death_age.copy()
df_death_age.head()

df_death_age.info()         # dane prawie kompletne, brak kodów nazw krajów do niektórych państw

df_psychopathology_age=pd.merge(df_depr_age, df_death_age, how='inner', on=['Entity', 'Year', 'Code'])
df_psychopathology_age.rename({'Entity': 'country','Code':'code', 'Year': 'year'}, axis=1, inplace=True) 
df_psychopathology_age.head()

df_psychopathology_age=pd.merge(df_psychopathology_age, df_psychopathology[['country', 'year', 'population']], how='inner', on=['country', 'year'])
df_psychopathology_age.head()

df_psychopathology_age.info()

df_psychopathology_age.rename({'Prevalence - Depressive disorders - Sex: Both - Age: 10 to 14 (Percent)':'depressive disorder (10-14)',    
                               'Prevalence - Depressive disorders - Sex: Both - Age: All Ages (Percent)':'depressive disorder (all ages)',
                               'Prevalence - Depressive disorders - Sex: Both - Age: 70+ years (Percent)':'depressive disorder (70+)',
                               'Prevalence - Depressive disorders - Sex: Both - Age: 50-69 years (Percent)':'depressive disorder (50-69)',
                               'Prevalence - Depressive disorders - Sex: Both - Age: 15-49 years (Percent)':'depressive disorder (15-49)'}, axis=1, inplace=True)

for col in df_psychopathology_age.columns[8:13]:                                                                              # zamieniam rate na procent
    df_psychopathology_age[col]=(df_psychopathology_age[col]*100000/df_psychopathology_age['population']).round(decimals=3)     
df_psychopathology_age.head()

df_psychopathology_age.rename({'Deaths - Self-harm - Sex: Both - Age: 70+ years (Rate)':'death - self harm (70+)',    
                               'Deaths - Self-harm - Sex: Both - Age: 50-69 years (Rate)':'death - self harm (50-69)',
                               'Deaths - Self-harm - Sex: Both - Age: All Ages (Rate)':'death - self harm (all ages)',
                               'Deaths - Self-harm - Sex: Both - Age: 5-14 years (Rate)':'death - self harm  (5-14)',
                               'Deaths - Self-harm - Sex: Both - Age: 15-49 years (Rate)':'death - self harm (15-49)'}, axis=1, inplace=True)

df_psychopathology_age.head()

df_psychopathology_age['year'].min(), df_psychopathology_age['year'].max(), len(df_psychopathology_age['country'].unique())

"""#Eksploracyjna analiza danych

"""

df_all.loc[:, 'life ladder':].describe().round(decimals=3)

columns_desstat=['life ladder', 'gdp per capita', 'healthy life expectancy at birth', 'schizophrenia', 'bipolar disorder', 'eating disorders', 'anxiety disorders', 
                 'drug use disorders', 'depressive disorders', 'alcohol use disorders', 'female suicide rate', 'male suicide rate', 'deaths by self-harm rate (both sexes)',
                 'male depressive disorder (percent)','female depressive disorder (percent)', 'population']

ax.set_title(column, fontsize='small', loc='left')
 fig.set_size_inches(14, 10)

fig, axes = plt.subplots(2,8, figsize=(20,6))

for i, col in enumerate(list(df_all[columns_desstat].columns.values)):
  a = df_all.boxplot(col, ax=axes.flatten()[i], fontsize='small')

plt.tight_layout() 
plt.show()

q1=df_all[columns_desstat].quantile(0.25)
q3=df_all[columns_desstat].quantile(0.75)
iqr=q3-q1

((df_all[columns_desstat]<(q1-1.5*iqr))|(df_all[columns_desstat]>(q3+1.5*iqr))).sum()

df_eda=pd.DataFrame()
df_eda['skew']=df_all[columns_desstat].skew().round(decimals=3)
df_eda['kurtosis']=df_all[columns_desstat].kurtosis().round(decimals=3)
df_eda['variance']=df_all[columns_desstat].var().round(decimals=3)
df_eda

fig, axes = plt.subplots(2,8)

for i, el in enumerate(list(df_all[columns_desstat].columns.values)):
  a = df_all.hist(el, ax=axes.flatten()[i])
  
fig.set_size_inches(27, 9)
plt.tight_layout() 
plt.show()

sns.set(font_scale=1.0)                                                                                                                                                              #rozkład dwumodalny
sns.distplot(df_all['life ladder'], color='forestgreen', axlabel='poziom satysfakcji życiowej').set(title='Histogram poziomu satysfakcji życiowej dla wszystkich krajów')
plt.ylabel('częstość występowania')

df_all3=df_all.copy()                                                                                                                          

df_all3['gdp per capita']=df_all3['gdp per capita'].ffill()
df_all3.info()

columns_desstat2=['life ladder', 'gdp per capita', 'healthy life expectancy at birth', 'schizophrenia', 'bipolar disorder', 'eating disorders', 'anxiety disorders', 
                 'drug use disorders', 'depressive disorders', 'alcohol use disorders', 'deaths by self-harm rate (both sexes)']    #bez populacji

for col in df_all3[columns_desstat2]:           
  print(col,':', shapiro(df_all3[col]))

"""Dla zmiennej *gdp per capita* p-value jest mniejsza od 0,05, zatem rozkład zmiennej nie jest normalny. Wartość p zmiennej *life ladder* (oraz pozostałych) przekracza wartość 0,05, zatem brak podstaw do stwierdzenia, że dane nie pochodzą z rozkładu normalnego."""

df_all2=df_all.copy()

df_box_cox=df_all2['gdp per capita']
trans_data, trans_lambda=stats.boxcox(df_box_cox)

sns.distplot(df_box_cox, hist=True, kde=True, kde_kws={'fill':True, 'linewidth':2}, color="forestgreen")

shapiro(df_box_cox)

df_all3['gdp per capita']=np.log(df_all['gdp per capita'])

sns.set(font_scale=1.0)                                                                                                                                               
sns.distplot(df_all3['gdp per capita'], bins=14, color='forestgreen', axlabel='GPD per capita').set(title='Histogram GDP per capita dla wszystkich krajów')
plt.ylabel('częstość występowania')

shapiro(df_all3['gdp per capita'])              # histogram sugeruje rozkład trzymodalny ale wynik Shapiro-Wilka nie daje podstaw do odrzucenia hipotezy o normalności testu........?

"""Po nałożeniu logarytmu  wartość p zmiennej *gdp per capita* przekracza wartość 0.05, zatem brak podstaw do stwierdzenia, że dane nie pochodzą z rozkładu normalnego."""

sns.pairplot(df_all[['life ladder', 'gdp per capita', 'healthy life expectancy at birth', 'schizophrenia', 'bipolar disorder', 'eating disorders', 'anxiety disorders', 'drug use disorders', 'depressive disorders', 'alcohol use disorders', 
                     'female suicide rate', 'male suicide rate', 'deaths by self-harm rate (both sexes)', 'male depressive disorder (percent)', 'female depressive disorder (percent)']], height=2.2)
plt.show()

"""W następnym kroku przyjrzę się zależnościom między zmiennymi: obliczę współczynniki korelacji rang Spearmana oraz sprawdzę ich istotność statystyczną."""

plt.figure(figsize=(9, 8))   
corr_psychop=df_all3[columns_desstat2].corr(method = 'spearman').round(decimals=3)
mask=np.zeros_like(corr_psychop)
mask[np.triu_indices_from(mask)]=True
sns.heatmap(corr_psychop, cmap='Spectral_r', mask=mask, square=True, vmin=-.6, vmax=0.95, annot=True, annot_kws={"size":10}, linewidths=1.2, linecolor='white')
plt.tick_params(labelsize=10)
plt.title('Macierz korelacji rang Spearmana', fontsize=16, pad=13)

df_all3[columns_desstat2].rcorr(method='spearman')

"""Na początku przyjrzę się jak zmieniał się poziom poczucia szczęścia mieszkańców całego świata w latach 2007 - 2019. Następnie zidentyfikuję najmniej i najbardziej szczęśliwe narodowości w latach 2007 i 2019 oraz stworzę ranking krajów utworzony na podstawie średniego poziomu raportowanego poczucia szczęścia (uwzględniający wszystkie badane lata)."""

happiness_map=px.choropleth(df_all.sort_values('year'),                
              locations='alpha-3',
              color='life ladder', 
              hover_name='country',
              hover_data=['year', 'life ladder'],
              animation_frame='year',
              color_continuous_scale="YlGn",
              locationmode='ISO-3',
              scope='world',
              labels={'life ladder':'poziom<br>satysfakcji<br>życiowej'},
              range_color=(2.0, 8.2),
              width=1600,
              height=950)
happiness_map.update_layout(title_text='Poziom raportowanej satysfakcji życiowej na świecie w latach 2005-2019', font=dict(size=15), title_x=0.5)
happiness_map

to_agg={'life ladder': ['describe']}                                                # podstawowe statystyki opisowe dla zmiennej 'life ladder' dla poszczególnych krajów
df_all_describe=df_all.groupby(['country']).agg(to_agg).round(2)
df_all_describe.columns=df_all_describe.columns.droplevel([0, 1])
df_all_describe=df_all_describe.reset_index()
df_all_describe['range']=df_all_describe['max']-df_all_describe['min']  
df_all_describe

agg_des={'country':'count', 'life ladder':['mean', 'min', 'median', 'max', 'std', 'var']}           # podstawowe statystyki opisowe dla zmiennej 'life ladder' dla poszczególnych lat (wraz z liczebą krajów, których dane zostały zagregowane)
df_all_year=df_all.groupby(['year']).agg(agg_des).round(3)
df_all_year.columns=df_all_year.columns.droplevel(0)
df_all_year=df_all_year.reset_index()
df_all_year['range']=df_all_year['max']-df_all_year['min']
df_all_year

"""Jak wynika z tabeli, z upływającymi latami obserwujemy wzrost zmienności poziomu satysfakcji życiowej (wyrażony poprzez rozstęp)."""

plt.rcParams['figure.figsize']=(10, 5)                                                              
plt.plot(df_all_year['year'], df_all_year['max'], color='forestgreen')
plt.plot(df_all_year['year'], df_all_year['mean'], linewidth=3, color='gold')
plt.plot(df_all_year['year'], df_all_year['median'], linewidth=3, color='darkorange')
plt.plot(df_all_year['year'], df_all_year['min'], color='red')
plt.title('Zmiany poziomu satysfakcji życiowej w czasie', fontsize=13, pad=18)
plt.yticks(np.arange(2.5, 8.5, 0.5), fontsize=12)
plt.grid(axis='y', linestyle = '--', linewidth = 0.7)
plt.xlabel('rok', fontsize=12, labelpad=10)
plt.ylabel('poziom satysfakcji życiowej', fontsize=12, labelpad=10)
plt.legend(['maksymalny poziom', 'średni poziom', 'środkowy poziom', 'minimalny poziom'], prop = {'size' : 10}, bbox_to_anchor=(1.35, 0.5), loc='center right', frameon=False)
plt.show()

"""Średni poziom satysfakcji życiowej od 2006 roku zaczyna utrzymuje się na mniej więcej stałym poziomie, z niewielką tendencją wzrostową od 2016 roku. Obserowowany spadek średniej w latach 2005-2006 zapewne wynika z niekompletnych danych, co można zaoważyć dzięki analizie kartogramu:

*   w 2005 roku nie zostały zebrane dane dla dwóch regionów świata (Afryka, Południowa Azja), w których przeciętny poziom satysfkacji utrzymuje się na wyraźnie niższym poziomie niż najszczęśliwszych rejonów świata (Europa, Ameryka Północna, Australia), stąd najwyższa (w obserowanym okresie) średnia tej zmiennej;
*   rak danych z części krajów europejskich i Autralii, spowodował spadek średniej, a pojawienie się  informacji z Afryki i Południowej Azji jeszcze pogłębiło ów spadek;
* wzrost średniej satysfakcji życiowej w 2007 wynika z pojawienia się kompletnych danych z Australii, Ameryki Północnej, większości danych z Europy.


"""

mean_top=df_all_describe.sort_values('mean', ascending=False).head(20)         
mean_bottom=df_all_describe.sort_values('mean').head(20)       
countries=list(mean_top['country'])
countries_bottom=list(mean_bottom['country'])

a=df_all[df_all['country'].isin(list(mean_top['country']))]
b=df_all[df_all['country'].isin(list(mean_bottom['country']))]

fig, (ax1, ax2)=plt.subplots(1, 2, figsize=(12, 9))

mean_top[['mean', 'min', 'max']].sort_values('mean', ascending=True).plot(kind='barh', width = 0.7, ax=ax1, colormap='Accent', fontsize=10)
ax1.set_xlabel('Poziom satysfakcji życiowej', fontsize=10, labelpad=12)
ax1.set_title('20 państw z największą średnią\nsatysfakcją życia mieszkańców', fontsize=12, pad=20)
ax1.set_yticklabels(countries)
ax1.get_legend().remove()

mean_bottom[['mean', 'min', 'max']].sort_values('mean', ascending=False).plot(kind='barh', width = 0.7, ax=ax2, colormap='Accent', fontsize=10)
ax2.set_xlabel('Poziom satysfakcji życiowej', fontsize=10, labelpad=12)
ax2.set_title('20 państw z najmniejszą średnią\nsatysfakcją życia mieszkańców', fontsize=13, pad=20)
ax2.set_yticklabels(countries_bottom)
ax2.get_legend().remove()

fig.subplots_adjust(wspace=0.8)
fig.legend(['średnia', 'min', 'max'], prop = {'size' : 10}, loc='center right', frameon=False)
fig.show()

df_2007_happiness=df_all.query("year==2007")
df_2007_the_happiest=df_2007_happiness.sort_values('life ladder', ascending=False).head(20)
df_2007_the_unhappiest=df_2007_happiness.sort_values('life ladder').head(20)

df_2019_happiness=df_all.query("year==2019")
df_2019_the_happiest=df_2019_happiness.sort_values('life ladder', ascending=False).head(20)    
df_2019_the_unhappiest=df_2019_happiness.sort_values('life ladder').head(20)

b=df_all[df_all['country'].isin(list(df_2007_the_happiest['country']))]
c=df_all[df_all['country'].isin(list(df_2007_the_unhappiest['country']))]

d=df_all[df_all['country'].isin(list(df_2019_the_happiest['country']))]
e=df_all[df_all['country'].isin(list(df_2019_the_unhappiest['country']))]

fig2007=pd.concat([b, c], ignore_index=True)
fig2019=pd.concat([d, e], ignore_index=True)

figura=px.line(x=fig2007['year'], y=fig2007['life ladder'], color=fig2007['country'], hover_name=fig2007['country'])
figura.update_layout(title='Zmiany poziomu satysfakcji życiowej w czasie<br>w krajach o najwyższym i najniższym poziomie tej cechy w 2007 roku',
                    title_x=0.45,
                    title_font_size=22,
                    #title_pad=12,
                    xaxis_title='rok',
                    yaxis_title='poziom satysfakcji życiowej',
                    legend_title='',
                    legend_font_size=12,
                    legend_y=0.5,
                    font=dict(size=14),
                    hoverlabel=dict(bgcolor="white", font_size=14),
                    width=1600,
                    height=800)
figura.update_traces(hovertemplate='rok: %{x}, poziom satysfakcji życiowej: %{y}')
figura.add_hline(y=5.45, line_width=2, line_dash="dash", line_color="darkorange")   
figura.show()

figura=px.line(x=fig2019['year'], y=fig2019['life ladder'], color=fig2019['country'], hover_name=fig2019['country'])
figura.update_layout(title='Zmiany poziomu satysfakcji życiowej w czasie<br>w krajach o najwyższym i najniższym poziomie tej cechy w 2019 roku',
                    title_x=0.45,
                    title_font_size=22,
                    #title_pad=12,
                    xaxis_title='rok',
                    yaxis_title='poziom satysfakcji życiowej',
                    legend_title='',
                    legend_font_size=12,
                    legend_y=0.5,
                    font=dict(size=14),
                    hoverlabel=dict(bgcolor="white", font_size=14),
                    width=1600,
                    height=800)
figura.update_traces(hovertemplate='rok: %{x}, poziom satysfakcji życiowej: %{y}')
figura.add_hline(y=5.45, line_width=2, line_dash="dash", line_color="darkorange") 
figura.show()

bar=go.Bar(x=df_2007_the_happiest['country'],
           y=df_2007_the_happiest['life ladder'],
           name='2007', 
           marker_color='lightseagreen')

bar1=go.Bar(x=df_2019_the_happiest['country'],
            y=df_2019_the_happiest['life ladder'],
            name='2019',
            marker_color='forestgreen')

data=[bar, bar1]       
plotly.offline.iplot({'data':data,
                      'layout':go.Layout(barmode='group', 
                                         title="Najbardziej szczęśliwe kraje świata w latach 2007 i 2019",
                                         title_x=0.5,
                                         title_font=dict(size=20),
                                         legend_font=dict(size=16),
                                         width=1500, height=550)})

bar=go.Bar(x=df_2007_the_unhappiest['country'],
           y=df_2007_the_unhappiest['life ladder'],
           name='2007',
           marker_color='lightseagreen')

bar1=go.Bar(x=df_2019_the_unhappiest['country'],
            y=df_2019_the_unhappiest['life ladder'],
            name='2019',
            marker_color='forestgreen')

data=[bar, bar1]
plotly.offline.iplot({"data": data,
                      "layout": go.Layout(barmode='group', 
                                          title="Najmniej szczęśliwe kraje świata w latach 2007 i 2019 - TOP 20",
                                          title_x=0.5,
                                          title_font=dict(size=20),
                                          legend_font=dict(size=16),
                                          width=1500, height=550)})

"""PODSUMOWANIE POWYŻSZYCH:..................."""

df_all_range=df_all_describe[['country', 'range']].sort_values('range')  
df_all_range=df_all_range[df_all_range['range']> 0]

fig=go.Figure()
fig.update_layout(barmode='group', title_text='Kraje o najmniejszym i największym zakresie zmienności poziomu satysfakcji życiowej w całym badanym okresie')
fig.add_trace(go.Bar(x=df_all_range['country'].head(20), y=df_all_range['range'].head(20)))
fig.add_trace(go.Bar(x=df_all_range['country'][-20:], y=df_all_range['range'][-20:]))

df_change, df_change2=(pd.DataFrame(), pd.DataFrame())
for index, row in df_2007_happiness.iterrows():
  df_change['country']=df_2007_happiness['country']
  df_change['life ladder 2007']=df_2007_happiness['life ladder']

for index, row in df_2019_happiness.iterrows():
  df_change2['country']=df_2019_happiness['country']
  df_change2['life ladder 2019']=df_2019_happiness['life ladder']

df_diff=df_change.merge(df_change2, on='country', how='inner')
df_diff['life ladder change']=df_diff['life ladder 2019']-df_diff['life ladder 2007']
df_diff.sort_values('life ladder change', inplace=True)

fig = go.Figure()
fig.update_layout(barmode='group', title_text='Największe zmiany w poziomie satysfakcji życiowej w porównaniu lat 2007 i 2019')
fig.add_trace(go.Bar(x=df_diff['country'].head(20), y=df_diff['life ladder change'].head(20)))
fig.add_trace(go.Bar(x=df_diff['country'][-20:], y=df_diff['life ladder change'][-20:]))

df_all4=df_all3.groupby('country').mean().reset_index()
df_all4.head()

figg=px.scatter(df_all4, x='gdp per capita', y='life ladder', size=list(df_all4['population']), size_max=75, hover_name=df_all4['country'], 
                title='Związek między wysokością PKB na mieszkańca a poziomem satysfakcji życiowej<br>dla wszystkich krajów świata (z uwzględnieniem wielkości populacji)', 
                color_continuous_scale=px.colors.sequential.Viridis, range_color=[0.05,1.4], height=600, width=1600)
figg.update_layout(legend=dict(x=0, y=1, font=dict(size=12), bgcolor="white", bordercolor="white"),
                   title=dict(font_size=20, x=0.45))
figg.update_xaxes(title='PKB na mieszkańca (USD)')
figg.update_yaxes(title='poziom satysfakcji życiowej')
figg.show()

"""### REGRESJA LINIOWA

na zlogarytmowanych danych GDP
"""



df_all5=df_all3.copy()

index_gdp=df_all5[df_all5['gdp per capita'].isnull()].index
df_all5.drop(index_gdp, inplace=True)

lin_reg=pg.linear_regression(df_all5['gdp per capita'], df_all5['life ladder'])        # przewidywanie satysfakcji życiowej na postawie GDP per capita
lin_reg.round(2)

"""*GDP per capita* jest istotnym statystycznie (T = 57.68, p-value < 0.05) predyktorem satysfakcji życiowej. Współczynnik determinacji (tak jak skorygowany współczynnik determinacji) wynosi 0.67, co znaczy, że stworzony model regresji wyjaśnia około 67% zmienności satysfakcji życiowej. """

corr_l_g=pg.corr(df_all5['gdp per capita'], df_all5['life ladder'], method='spearman')
sns.regplot(x='gdp per capita', y='life ladder', data=df_all5)
plt.show()

corr_l_g

pg.homoscedasticity(df_all5[['life ladder', 'gdp per capita']], method="bartlett", alpha=.05)



#--------------------------------

depression_map=px.choropleth(df_psychopathology_age.sort_values('year'),                
              locations='code',
              color='depressive disorder (15-49)', 
              hover_name='country',
              hover_data=['year', 'depressive disorder (15-49)'],
              animation_frame='year',
              color_continuous_scale="YlGnBu",
              locationmode='ISO-3',
              scope='world',
              labels={'depressive disorder (15-49)':'procent mieszkanców<br>chorych na depresję'},
              range_color=(2, 10.8),
              width=1600,
              height=950)
depression_map.update_layout(title_text='Częstoliwość występowania depresji na świecie w latach 1990-2019', font=dict(size=15), title_x=0.5)
depression_map

k=df_psychopathology_age[['year', 'depressive disorder (10-14)', 'depressive disorder (15-49)', 'depressive disorder (50-69)', 'depressive disorder (70+)', 'depressive disorder (all ages)']].groupby('year').mean().reset_index()

plt.rcParams['figure.figsize']=(10, 5)       
plt.plot(k['year'], k['depressive disorder (10-14)'], linewidth=2, color='lightsteelblue')                                                       
plt.plot(k['year'], k['depressive disorder (15-49)'], linewidth=2, color='royalblue')
plt.plot(k['year'], k['depressive disorder (50-69)'], linewidth=2, color='dodgerblue')
plt.plot(k['year'], k['depressive disorder (70+)'], linewidth=2, color='mediumturquoise')
plt.plot(k['year'], k['depressive disorder (all ages)'], linewidth=2, color='darkorange')
plt.title('Zmiany w czasie częstotliwości występowania depresji dla różnych grup wiekowych', fontsize=13, pad=18)
plt.yticks(np.arange(1.0, 7.0, 0.5), fontsize=10)
plt.grid(axis='y', linestyle = '--', linewidth = 0.7)
plt.xlabel('rok', fontsize=10, labelpad=10)
plt.ylabel('częstotliwość występowania depresji', fontsize=10, labelpad=10)
plt.legend(['10 - 14', '15 - 49', '50 - 69', '70+', 'wszystkie grupy wiekowe'], prop = {'size' : 10}, bbox_to_anchor=(1.35, 0.5), loc='center right', frameon=False)
plt.show()

"""Sprawdzam, czy między wartościami dla badanych grupa istnieje istotna statystyczna różnica."""

columns_age=['depressive disorder (10-14)', 'depressive disorder (all ages)', 'depressive disorder (70+)', 'depressive disorder (50-69)', 'depressive disorder (15-49)']

for col in df_psychopathology_age[columns_age]:           # wyłączam z analizy grupę wiekową 10-14, ponieważ wymaganiem testu ANOVA są normalne rozkłady badanych wartości
  print(col,':', shapiro(df_psychopathology_age[col]))

pg.homoscedasticity(df_psychopathology_age[['depressive disorder (all ages)', 'depressive disorder (70+)',               # wartość p-value wkazuje na potrzebę skorzystania z testu Welcha 
       'depressive disorder (50-69)', 'depressive disorder (15-49)']], method="bartlett", alpha=.05)

df_anova=pd.melt(df_psychopathology_age[['country', 'depressive disorder (all ages)', 'depressive disorder (70+)',
       'depressive disorder (50-69)', 'depressive disorder (15-49)']], id_vars='country')
df_anova.drop(['country'], axis=1, inplace=True)
df_anova.rename(columns={df_anova.columns[0]: 'grupa wiekowa', df_anova.columns[1]: 'częstotliwość'},inplace=True)

aov = welch_anova(dv='częstotliwość', between='grupa wiekowa', data=df_anova)
aov

pg.pairwise_gameshowell(data=df_anova, dv='częstotliwość',
                        between='grupa wiekowa').round(3)

"""Wyniki powyższych testów statystycznych wskazują iż:

*   częstotliwości występowania depresji w wybranych grupach wiekowych są heteroskedastyczne,
*   badane grupy wiekowe różnią się między sobą w sposób istotny statycznie oprócz pary wiekowej 50-69 lat i 70+.



"""
