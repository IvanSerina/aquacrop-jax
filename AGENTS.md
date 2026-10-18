# AquaCrop JAX — coordinatore locale

Obiettivo: completare il porting Fortran → JAX mantenendo input, output e
comportamento numerico compatibili.

## Componenti da considerare

Il coordinatore deve mantenere una visione dell'intero modello:

1. parsing di progetti e dati climatici;
2. temperatura e growing degree days;
3. profilo del suolo, falda e capacità di campo corretta;
4. drenaggio, runoff, infiltrazione e storage superficiale;
5. risalita capillare;
6. salinità, trasporto e concentrazione dei sali;
7. germinazione, fenologia e radici;
8. irrigazione e gestione colturale;
9. stress idrico, salino, termico, anaerobico e di fertilità;
10. canopy cover e relativa retroazione;
11. evaporazione e traspirazione;
12. biomassa, assimilati, harvest index, tagli e resa;
13. bilanci giornalieri di acqua e sale;
14. runner completo, file `.OUT`, statistiche e compatibilità CLI;
15. precisione `float64` di riferimento e riproducibilità su CPU. GPU e altri
    acceleratori sono esclusi quando è attivo `PROMPT_CLOUD_CPU.md`;
16. prestazioni, compilazione JIT, batching e riproducibilità.

## Ciclo di lavoro

1. Leggere `.sequence3/NEXT_RUN.md`, se presente, prima di modificare codice.
2. Scegliere una sola unità funzionale e mappare la routine Fortran sorgente.
3. Aggiungere o aggiornare fixture di parità e test mirati.
4. Implementare la modifica JAX senza indebolire i test esistenti.
5. Eseguire prima i test mirati tramite `tools/test-and-capture`.
6. Se i test mirati passano, continuare con il prossimo incremento.
7. Se un test fallisce, `tools/test-and-capture` esegue automaticamente
   `tools/failure-gate`; non interrompere questo gate diagnostico.
8. Per il workflow CPU-only di questo repository non chiamare `tools/ask-cloud`.
   Per esplorazione e lavoro meccanico usa `tools/delegate-local` quando il
   servizio CPU locale è disponibile; se manca, registralo in `NEXT_RUN.md` e
   prosegui con gli incrementi numerici verificabili.

## Quando sospendere un incremento numerico

Non formulare una correzione congetturale quando almeno una condizione è vera:

- lo stesso errore resiste a due tentativi locali;
- la divergenza Fortran/JAX non ha una causa dimostrata;
- cambia una ricorrenza o un bilancio conservativo;
- cambia salinità, biomassa, harvest index, resa o logica degli eventi;
- serve la review finale di una componente.

Confrontare sorgente e oggetti Fortran e aggiungere una fixture mirata. Se la
review prevista non è disponibile, salvare in `NEXT_RUN.md` evidenze, tentativi
e prossimo passo senza indebolire tolleranze né dichiarare parità.

## Regola in caso di fallimento

Ogni fallimento deve lasciare materiale sufficiente per la sessione seguente:

- log del test mirato;
- suite completa `pytest`;
- generazione degli stadi JAX disponibili;
- confronto con gli `.OUT` Fortran, se materialmente disponibile;
- benchmark JAX e Fortran, se gli eseguibili sono disponibili;
- stato, diff e ultimo commit Git;
- ambiente Python/JAX/backend;
- diagnosi, tentativi già eseguiti e prossimo passo in `NEXT_RUN.md`.

Non dichiarare parità sulla sola base dei test unitari. La parità completa
richiede runner stagionale, stessi file di input, stessi `.OUT` e tolleranze
documentate campo per campo.

## Git

Non riscrivere la cronologia né usare reset distruttivi. Creare checkpoint
descrittivi dopo gli incrementi verificati quando l'utente ha chiesto commit
periodici oppure `AQUACROP_AUTOCOMMIT=1`; altrimenti conservare stato e diff nel
dossier. Aggiungere solo file appartenenti all'incremento: non includere
`.sequence3/` o file preesistenti non tracciati.
