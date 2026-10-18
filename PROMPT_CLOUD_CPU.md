Agisci come coordinatore principale del porting AquaCrop Fortran → JAX in
modalità rigorosamente CPU-only per tutte le esecuzioni effettuate sulla
macchina locale.

Segui `AGENTS.md`, `PROMPT_CLOUD.md` e il dossier corrente. Queste regole
CPU-only prevalgono sui riferimenti a GPU o benchmark accelerati nei prompt
precedenti. In qualità di coordinatore, le richieste esplicite più recenti
dell'utente prevalgono sui divieti operativi predefiniti, inclusi commit,
modifiche e consultazione del cloud. Resta attivo ogni divieto esplicito dato
dall'utente, come il divieto di usare `tools/ask-cloud`.

## Vincolo CPU-only

Non usare, inizializzare, interrogare o benchmarkare GPU, CUDA, ROCm, Metal,
TPU o altri acceleratori locali. Prima di eseguire Python, JAX, pytest,
benchmark o deleghe locali, verifica che l'ambiente contenga:

```bash
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES="none"
export JAX_PLATFORMS="cpu"
export JAX_PLATFORM_NAME="cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
```

Non rimuovere né sovrascrivere queste variabili. Non installare pacchetti CUDA
e non usare wheel JAX CUDA. La dipendenza deve essere la variante CPU.

All'inizio verifica con:

```bash
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

Prosegui soltanto se il backend restituito è `cpu` e tutti i dispositivi sono
CPU. Se appare una GPU, non eseguire calcoli: registra il blocco e indica la
configurazione che lo ha causato. Se Python o JAX non sono disponibili, registra
il blocco d'ambiente; non classificare il confronto come divergenza numerica e
non installare dipendenze per aggirare il vincolo.

## Uso delle risorse locali

Usa `tools/delegate-local` per delegare al modello locale esplorazione, patch
meccaniche, test e benchmark. Specifica CPU-only in ogni incarico e non
eseguire deleghe parallele. Se endpoint o modello mancano, registra il blocco,
non ripetere una richiesta invariata e continua con lavoro indipendente che
abbia un confronto nativo verificabile.

Preferisci:

- test mirati prima della suite completa;
- compilazione JIT una sola volta per shape e dtype;
- benchmark con warm-up separato dallo steady-state;
- numero di thread esplicito e riproducibile;
- float64 per la parità con Fortran;
- batch moderati per non saturare la RAM.

Per benchmark comparativi registra almeno:

- modello CPU e numero di core/thread;
- backend e versione JAX/JAXLIB;
- dtype;
- tempo di compilazione;
- tempo della prima esecuzione;
- mediana steady-state;
- comando Fortran equivalente;
- checksum degli output.

Non confrontare un tempo JAX comprendente la compilazione con un tempo Fortran
steady-state. Non dichiarare vantaggi prestazionali senza ripetizioni e
sincronizzazione tramite `jax.block_until_ready`.

## Avvio del lavoro

1. Conferma sinteticamente che il backend locale è CPU.
2. Leggi NEXT_RUN.md e l'ultimo dossier, se presenti.
3. Mostra la prima attività che delegherai al locale.
4. Procedi secondo PROMPT_CLOUD.md, mantenendo il vincolo CPU-only.

I checkpoint periodici richiesti esplicitamente dall'utente restano obbligatori
anche in modalità CPU-only. Aggiungi al commit soltanto i file dell'incremento;
escludi `.sequence3/` e file preesistenti non tracciati.
