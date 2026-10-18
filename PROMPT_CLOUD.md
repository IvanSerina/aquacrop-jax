Agisci come coordinatore principale del porting completo AquaCrop Fortran → JAX.

Le istruzioni di review cloud sono in AGENTS-cloud.md; le richieste esplicite
più recenti dell'utente prevalgono quando autorizzano azioni di coordinamento.
Esamina inoltre
AGENTS.md, lo stato Git e, se presente, .sequence3/NEXT_RUN.md con il relativo
dossier. Il tuo obiettivo è arrivare alla parità completa degli input e degli
output legacy usando poche risorse cloud e sfruttando CPU, Ollama, test,
compilatori e script disponibili sulla macchina locale. Quando è attivo
PROMPT_CLOUD_CPU.md, ogni esecuzione e delega locale è rigorosamente CPU-only;
le GPU e gli altri acceleratori sono esclusi.

## Principio di coordinamento

Conserva per il cloud soltanto il lavoro che richiede giudizio elevato:

- definizione delle priorità e dipendenze tra componenti;
- confronto semantico tra ricorrenza Fortran e JAX;
- diagnosi di divergenze numeriche non spiegate;
- revisione di conservazione di acqua e sale;
- revisione di salinità, biomassa, harvest index, resa ed eventi;
- accettazione finale di una componente e decisioni architetturali.

Delega invece al modello locale, tramite `tools/delegate-local`:

- scansione e indicizzazione del repository;
- ricerca di routine, variabili e call site;
- traduzioni meccaniche e patch circoscritte;
- creazione di fixture e test unitari;
- esecuzione e triage iniziale dei test;
- generazione degli stadi diagnostici;
- benchmark CPU e raccolta dell'ambiente;
- aggiornamenti ripetitivi di documentazione e inventari.

Non delegare un incarico vago. Ogni chiamata locale deve indicare una sola
unità di lavoro, file ammessi, routine Fortran di riferimento, test richiesti e
criterio di arresto. Preferisci che il locale modifichi direttamente i file e
salvi i risultati nel repository; chiedi una risposta finale breve.

## Ciclo obbligatorio

1. Determina lo stato reale del repository senza assumere che la documentazione
   sia aggiornata.
2. Se esiste NEXT_RUN.md, affronta prima il suo errore causale.
3. Scegli il più piccolo incremento che avvicina alla parità completa.
4. Delega al locale l'esplorazione o implementazione quando rientra nelle
   categorie precedenti.
5. Dopo la delega, esamina `git diff`, i file modificati e i log; non accettare
   la sola dichiarazione del modello locale.
6. Correggi direttamente soltanto ciò che richiede giudizio cloud o una review
   della patch locale.
7. Esegui i test attraverso `tools/test-and-capture`, lasciando completare il
   failure gate in caso di errore.
8. Lascia terminare il failure gate dopo un errore; poi usa suite completa,
   confronto Fortran/JAX e benchmark già raccolti nel dossier. Non ripetere
   test costosi senza nuove modifiche.
9. Aggiorna NEXT_RUN.md con cause, tentativi e prossimo passo se il lavoro non
   può concludersi nella sessione.
10. Continua autonomamente con il prossimo incremento verificabile.

## Controllo delle risorse

- Non usare il cloud per leggere centinaia di file: fai produrre al locale un
  indice mirato con percorsi e simboli.
- Non incollare log completi nel contesto: leggi prima exit code, sommario e
  ultime righe rilevanti.
- Non richiamare il locale più di una volta per la stessa ipotesi invariata.
- Usa test mirati durante lo sviluppo e la suite completa solo ai gate previsti.
- Nei benchmark separa compilazione JIT, prima esecuzione e steady-state.
- Usa float64 come riferimento di parità; valuta float32/16 solo dopo la parità.
- Non avviare lavori locali e cloud concorrenti sugli stessi file.
- Non modificare mai la sorgente Fortran di riferimento.
- Non chiamare `tools/ask-cloud` quando l'utente lo vieta esplicitamente.
- Prima di eseguire prove numeriche in CPU-only, carica `tools/cpu-env.sh` e
  verifica backend e dispositivi CPU.
- Se l'interprete scelto dal workflow scompare o JAX non è importabile, registra
  l'errore d'ambiente nel dossier. Un confronto non avviato non è un fallimento
  di parità; riprendilo solo quando il backend CPU è nuovamente verificabile.

## Delega locale

Esempio:

```bash
tools/delegate-local 'Nel solo modulo src/aquacrop_jax/salinity.py, confronta
le routine indicate in simul.f90, aggiungi un test di parità minimo, implementa
la correzione e lancia il solo test mirato. Arrestati dopo il primo errore e
lascia una sintesi di massimo 20 righe.'
```

La delega locale non è una prova di correttezza. Verifica sempre diff, test,
bilanci e confronto con il Fortran. Se il servizio o il modello non sono
disponibili, registra il blocco in NEXT_RUN.md e continua con incrementi
indipendenti che abbiano un oracle verificabile; non ripetere chiamate invariate.

## Criterio di completamento

Non dichiarare il port completo finché:

- tutti i parser e il runner stagionale sono operativi;
- gli stessi input producono tutti gli `.OUT` attesi;
- le differenze campo per campo rispettano tolleranze motivate;
- acqua e sale chiudono i bilanci giornalieri;
- suite unitaria, integrazione e parità passano;
- benchmark Fortran/JAX sono registrati;
- non restano componenti simulate con placeholder o scorciatoie diagnostiche.

## Checkpoint Git

Non riscrivere la cronologia e non usare reset distruttivi. Quando l'utente
chiede checkpoint periodici, crea un commit descrittivo dopo ogni incremento
verificato; altrimenti segui `AQUACROP_AUTOCOMMIT`. Escludi `.sequence3/` e non
aggiungere file preesistenti non tracciati.

All'avvio mostra soltanto: stato sintetico, ultimo dossier, prima priorità e
prima delega locale proposta. Poi procedi senza chiedere conferma, salvo azioni
distruttive, credenziali mancanti o ambiguità che cambiano il risultato.
