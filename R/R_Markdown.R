#�ah�k : http://www.rstudio.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf

## Template css.

h1{
  color: white;
  padding: 10px;
  background-color: #3399ff
}

ul {
  list-style-type: square;
}

.MathJax_Display {
  padding: 0.5em;
  background-color: #eaeff3
}


### My_document.RMD


#####   #HEADER 1,  ## HEADER 2,  ### HEADER 3 ... zmen�uj� sa pod�a po�tu mrie�ok 
#####  2 mrie�ky a n�zov robi nadpis nejakej �asti
#####  ` ` ke� d�me nie�o medzi dva opa�n� apostrofi, vytvor�me v dokumente �ed� obd�nik okolo toho slova-vyzna�enie slova
#####  [2006 ASA Data Expo](http://stat-computing.org/dataexpo/2006/).  Vytvor� hypertext s labelom v [...] a odkazom na (...)
#####  * **temp** , prv� hviezdi�ka je len begin item (stvorcek na zarovnanie)  **text** je bold , *text* je italic
#####  *text*, alebo _text_ d�va italic a **text** bold
#####  $ text $ je klasick� latex, d� to do nov�ho riadku a cel� vzorec d� do �ed�ho �tvor�eka. Funguj� latex zna�enia \times,\frac
#####  <http://rmarkdown.rstudio.com>.  link s modrou �iarou pod textom

#####
## ```{r}
## summary(cars)
## ```
#####  toto sprav� to, �e summary(cars) bude v �edom �tvor�eku a navy�e pod t�m bude v�stup summary 

#### ak by sme dali {r, echo=FALSE} , tak k�d sa neuk�e, iba v�stup


--- 
  title: "Hello R Markdown"
output:
  html_document:   #pdf_document, word_document, beamer_presentation, slidy_presentation
  css: faded.css
---
  
  ## Data      
  
  The `atmos` data set resides in the `nasaweather` package of the *R* programming language. It contains a collection of atmospheric variables measured between 1995 and 2000 on a grid of 576 coordinates in the western hemisphere. The data set comes from the [2006 ASA Data Expo](http://stat-computing.org/dataexpo/2006/).

Some of the variables in the `atmos` data set are:
  
  * **temp** - The mean monthly air temperature near the surface of the Earth (measured in degrees kelvin (*K*))

* **pressure** - The mean monthly air pressure at the surface of the Earth (measured in millibars (*mb*))

* **ozone** - The mean monthly abundance of atmospheric ozone (measured in Dobson units (*DU*))

You can convert the temperature unit from Kelvin to Celsius with the formula

$$ celsius = kelvin - 273.15 $$
  
  And you can convert the result to Fahrenheit with the formula

$$ fahrenheit = celsius \times \frac{9}{5} + 32 $$
  
  ```{r, echo = FALSE, results = 'hide'}
example_kelvin <- 282.15
```

For example, `r example_kelvin` degrees Kelvin corresponds to `r example_kelvin - 273.15` degrees Celsius.

##################################################################################################################


# Rmarkdown nem� pr�stup do glob�lnych premenn�ch, v�dy pri renderovan� vytvor� nov� vlastn� enviroment. Vsetko
# na �o sa odvol�vas v k�de mus� by� definovan� v tom istom k�de

## Poznamka : aby sme predisli errorom messageom a podobne, treba pridat parametre 
```{r warning = FALSE, error = FALSE, message = FALSE}
"four" + "five"
```


#Three of the most popular chunk options are echo, eval and results.

# If echo = FALSE, R Markdown will not display the code in the final document 
# (but it will still run the code and display its results unless told otherwise).

# If eval = FALSE, R Markdown will not run the code or include its results, 
# (but it will still display the code unless told otherwise).

# If results = 'hide', R Markdown will not display the results of the code 
# (but it will still run the code and display the code itself unless told otherwise).

#fig.heigt, fig.width kontroluje velkos� obr�zku

```{r fig.height = 4, fig.width = 5}
means %>%
  ggvis(~temp, ~ozone) %>%
  layer_points()
```

#### Inline text:
## The factorial of four is `r factorial(4)`.  --> The factorial of four is 24.


#REFERNCIA + KOPIROVANIE KODU NA VIACERE MIESTA 
```{r simple_sum, results = 'hide'}  #simple_sum je label
2 + 2
```

```{r ref.label='simple_sum', echo = FALSE}   ## ref.label = "label" 
```

### Info: GGVIS je HTML object a nemoze byt pouzity do PDF, tam mus� �s� ggplot
### Render prikaz  :  rmarkdown::render(<file path>)   render any .Rmd file with R.



###########################
#### PREZENT�CIE ##########
###########################

---
  output: beamer_presentation       #which creates a beamer pdf slideshow
          #ioslides_presentation     which creates an ioslides HTML slideshow or
          #slidy_presentation        which creates a slidy HTML slideshow
---

  
### Slideshow:

---
  output: beamer_presentation
---
  
# R Markdown will start a new slide at each first or second level header in your document.
# You can insert additional slide breaks with Markdown's horizontal rule syntax:  ***
# Everywhere you add these three asterisks in your text, pandoc will create a new slide.
  
  
#http://www.rstudio.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf