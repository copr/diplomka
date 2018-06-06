Pro samotnou implementaci a testování jsme použili tyto verze Python interpreteru a knihoven
Python 3.4.3
scikit-learn==0.16.1
scipy==0.17.1
Theano==1.0.1
matplotlib==1.5.1
numpy==1.11.1

Pro vyzkoušení implementace algoritmu jsou připraveny skripty:
testRjmcmc.py - slouží jako ukázka algoritmu RJMCMC. Také slouží jako příklad jak používat námi vytvořene 			třídy.
testBlr2.py - Test MCMC algoritmu na lineární regresi s pevným počtem zlomů.

Důležité třídy:
blr2.py - implementace aposteriorního rozdělení pro bayesovskou lineární regresi
mcmc.py - implementace lehce upraveného algoritmu Metropoli Hastings
mcmc_kth.py - podobný jako mcmc.py, ale dá se nastavit, že každý ktý krok se použije jiná návrhová distribuce 
revers.py - zde je implementace RJMCMC algoritmu
move.py - reprezentuje jeden přeskok mezi dimenzemi a zpět

Dúležité skripty:
transformations.py - zde jsou zadefinovaný přechodové transformace pomocí theana a zde se také symbolicky
			spočítají jejich jakobiány.


