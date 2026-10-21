# Workflow locale CPU-only

Questi wrapper sono stati ricostruiti: non erano presenti nel checkout né nella
cronologia Git disponibile. Non cambiano i kernel JAX o il riferimento Fortran.

`cpu-env.sh` imposta i cinque vincoli CPU richiesti prima di avviare Python e
rifiuta valori preesistenti incompatibili senza sovrascriverli. Il preflight
verifica backend, dispositivi CPU e float64. Il Python scelto è quello della
`.venv`, se disponibile; `AQUACROP_PYTHON` consente una scelta esplicita.
`AQUACROP_CPU_THREADS` controlla i thread delle librerie numeriche (default: 1).

## Test e dossier

```bash
tools/test-and-capture -q tests/test_production.py
tools/test-and-capture -- .venv/bin/python -m pytest -q tests/test_workflow_tools.py
```

Ogni esecuzione conserva comando, exit code, ambiente e log in un dossier
univoco sotto `.sequence3/dossiers/`. In caso di errore, il wrapper attende il
completamento di `tools/failure-gate` e restituisce l'exit code del test originale,
anche quando il gate fallisce. Lo stato del gate è registrato separatamente.

Il gate raccoglie suite completa, quattro stadi diagnostici, confronto con gli
OUT disponibili, snapshot Git e copie di tools/src/tests, quindi aggiorna
`.sequence3/NEXT_RUN.md`. Il riferimento Fortran viene eseguito esclusivamente
su una copia isolata del testcase nel dossier. Le operazioni numeriche sono
sequenziali; gli errori di ogni comando rimangono nei rispettivi log.

Il benchmark JAX misura un kernel GDD float64 con lowering, compilazione,
prima esecuzione e sette misure sincronizzate separate. Il benchmark Fortran
misura il runner completo precompilato, con prima esecuzione e tre ripetizioni.
Gli ambiti sono diversi: **questi tempi non consentono confronti di velocità**.
Ambiente, comandi, parametri e checksum sono conservati nel dossier.

I wrapper non creano commit: il ruolo cloud resta read-only rispetto ai commit.
Lo snapshot conserva anche i sorgenti non tracciati di tools/src/tests.
Non usare `tools/git-checkpoint` per includere indiscriminatamente file
preesistenti non tracciati. `.sequence3/` è esclusa da Git.

## Delega locale

La configurazione predefinita locale è in `.env.local` (non tracciato): usa
Ollama sul loopback con `aquacrop-coder-14b:latest`, adatto alle analisi brevi
e ai task meccanici. Per cambiare modello modifica `AQUACROP_LOCAL_MODEL` con
il nome esatto restituito da `ollama list`; il wrapper non scarica modelli.

```bash
tools/delegate-local 'Unità: mappare il ramo HI per cereali e tuberi. \
Routine Fortran: LoadCrop e HarvestIndexDay. File ammessi: quelli allegati. \
Test richiesti: nessuna esecuzione, indicare solo le fixture mancanti. \
Arresto: mappa con percorsi/simboli e sintesi entro 20 righe.' \
  --file src/aquacrop_jax/crop_runner.py \
  --file src/aquacrop_jax/production.py \
  --file ../aquacrop-fortran/src/global.f90:4906:4917 \
  --file ../aquacrop-fortran/src/global.f90:5650:5747
```

Il wrapper usa l'API HTTP Ollama sul loopback (default
`http://127.0.0.1:11434`, configurabile con `AQUACROP_OLLAMA_URL`). Disabilita
proxy e redirect; non scarica modelli e non ripiega sul cloud. Non seleziona un
modello automaticamente. Una richiesta alla volta, massimo 128 KiB di file
di contesto, `num_gpu=0`, thread espliciti e scaricamento del modello dalla
memoria dopo la risposta (`keep_alive=0`).

Questa prima versione delega **analisi read-only sul contesto allegato**:
non concede accesso autonomo al filesystem e non applica patch o comandi
generati dal modello. Il coordinatore deve verificare la risposta. I file di
contesto devono appartenere al repository o al riferimento `aquacrop-fortran`.
La sintassi `percorso:prima_riga:ultima_riga` limita il contesto a un intervallo
con numeri di riga originali, senza allegare interi sorgenti Fortran.
Servizio o modello mancanti producono exit 69, dossier e prossimo passo.
