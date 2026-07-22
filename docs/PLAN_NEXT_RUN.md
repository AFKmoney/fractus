# PLAN DÉTAILLÉ — Prochaine session GPU (dans 9 jours)

## RÈGLE #1 : TESTER EN LOCAL AVANT DE TOUCHER LE POD

Chaque script, chaque modification, DOIT être testé en local sur ton PC avant d'aller sur le pod. Pas d'exception. Si je peux pas le tester en local, je le teste sur le pod avec un timeout de 60 secondes d'abord.

---

## ERREURS À NE PAS RÉPÉTER

### 1. Script tronqué (train_1b_cloud.py coupé à 197 lignes)
**Erreur :** Le script était incomplet, pas de main(), pas de opt.step()
**Fix permanent :** Toujours `python3 -c "import ast; ast.parse(open('script.py').read())"` + vérifier que `if __name__ == "__main__"` existe avant de lancer
**Status :** Fixé dans le code actuel

### 2. MoE detach bug (experts gelés)
**Erreur :** `moe_out.detach()` gelait les 64 experts
**Fix permanent :** Retiré du code. Testé : les experts reçoivent leurs gradients
**Status :** Fixé

### 3. Compteur de step bugué au resume
**Erreur :** Le compteur repartait de 0 au lieu de continuer
**Fix permanent :** Lit `step` du checkpoint, l'utilise comme start_step
**Status :** Fixé dans train_1b_cloud.py

### 4. Corpus corrompu (torch.save sur 64GB)
**Erreur :** torch.save ne gère pas les fichiers >50GB, corruption silencieuse
**Fix permanent :** Sauvegarder en .npy (numpy memmap), PAS en .pt
**Status :** À appliquer

### 5. Fusion shards qui OOM
**Erreur :** Charger 16B tokens en RAM pour les concaténer = OOM
**Fix permanent :** Lire les shards directement pendant le training, pas de fusion
**Status :** À coder (voir ci-dessous)

### 6. Conversion de format qui crash au dernier moment
**Erreur :** Passer 3 jours à préparer des données, crasher à la fin
**Fix permanent :** Le corpus build écrit des shards .npy. Le training lit les shards DIRECTEMENT. Pas de fusion. Pas de conversion. Pas de format intermédiaire.

---

## CE QUI DOIT ÊTRE PRÊT AVANT LE PROCHAINE POD

### A. Scripts (tous testés en local)

| Script | Status | Action requise |
|--------|--------|---------------|
| `edt_pipeline.py` | ✅ Validé sur GPU | Modifier pour lire shards .npy directement |
| `edt_pipeline.py` Phase 1 | ✅ Mesuré 1.2h | Utiliser VRAIS hidden states (pas synthétiques) |
| `edt_pipeline.py` Phase 2a | ✅ Mesuré <1s | OK |
| `edt_pipeline.py` Phase 2b | ✅ Mesuré 3.2h | OK |
| `edt_pipeline.py` Phase 3 | ✅ Mesuré 41h | OK |
| `build_fractus_1b_corpus.py` | ✅ Build fait 21B | Vérifier shards sur disque |
| `build_github_dataset.py` | ✅ 7.1M tokens | OK |
| `transfer_to_cte.py` | ✅ Validé | OK |
| `assemble_fractus.py` | ✅ Validé | OK |

### B. Corpus (21B tokens)

Le corpus est construit en 118 shards .npy sur le pod. **Mais le pod est détruit.** Il faut reconstruire.

**Option A :** Reconstruire sur le nouveau pod (~5-7h)
**Option B :** Pré-construire sur ton PC local (mais c'est long sans GPU)

**Reco :** Reconstruire sur le pod. Le script `build_fractus_1b_corpus.py` a du resume support — si le build crash, il reprend où il s'est arrêté.

### C. Auto-launch corrigé

Le script `auto_launch_edt.sh` doit :
1. Attendre que les shards existent (pas un fichier unique)
2. Lancer EDT qui lit les shards directement
3. Auto-push checkpoints sur HF

---

## ÉTAPE PAR ÉTAPE — PROCHAINE SESSION

### Étape 0 : Avant le pod (sur ton PC)

```bash
# 1. Pull le code le plus récent
cd ~/ZCodeProject/fractus-test
git pull

# 2. Vérifier que tous les scripts sont syntax-valid
python3 -c "import ast; [ast.parse(open(f).read()) for f in ['scripts/edt_pipeline.py', 'scripts/build_fractus_1b_corpus.py', 'scripts/build_github_dataset.py']]"
print("All scripts OK")

# 3. Vérifier que le modèle build à 1B
python3 -c "from fractus1B.model_1b import Fractus1B; m=Fractus1B(max_seq_len=32); print(f'{sum(p.numel() for p in m.parameters())/1e9:.3f}B')"
# Doit afficher: 1.049B
```

### Étape 1 : Lancer le pod

- Template : PyTorch
- GPU : RTX 3090 ou mieux (24GB+ VRAM)
- Disk : 300GB+ (corpus = 84GB)
- SSH : **Tester la connexion avant de continuer**

### Étape 2 : Setup (10 min)

```bash
apt-get install -y git gcc
pip install datasets zstandard transformers bitsandbytes

cd /workspace
git clone https://github.com/AFKmoney/fractus-test.git
cd fractus-test
```

### Étape 3 : Build corpus 21B (5-7h)

```bash
HF_TOKEN=<your_token> nohup python3 scripts/build_fractus_1b_corpus.py > corpus_build.log 2>&1 &
# Monitor: tail -f corpus_build.log
```

**NE PAS fusionner les shards en un fichier unique.** Garder les shards séparés.

### Étape 4 : Build GitHub dataset (30 min, en parallèle)

```bash
python3 scripts/build_github_dataset.py
# Ajoute les 7.1M tokens aux shards
```

### Étape 5 : Vérifier le corpus (1 min)

```bash
python3 -c "
import glob, numpy as np
shards = sorted(glob.glob('data/fractus_1b_shards/*_*.npy'))
total = sum(len(np.load(s, mmap_mode='r')) for s in shards)
print(f'{len(shards)} shards, {total/1e9:.2f}B tokens')
assert total > 15e9, 'Corpus trop petit!'
print('CORPUS OK')
"
```

### Étape 6 : Lancer EDT (2 jours)

```bash
HF_TOKEN=<your_token> \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/workspace/fractus-test \
nohup python3 scripts/edt_pipeline.py \
    --corpus data/fractus_1b_shards \
    --phase1-steps 2000 \
    --phase2a-steps 5000 \
    --embed-tokens 500000000 \
    --joint-tokens 100000000 \
    --batch-size 8 \
    --seq-len 32 \
    --save-every 5000 \
    > edt_training.log 2>&1 &

# Monitor: tail -f edt_training.log
```

**Le script EDT doit lire les shards directement, pas un fichier unique.** C'est la modification à faire avant la prochaine session.

### Étape 7 : Monitor

- Phase 1 (experts) : ~1.2h → checkpoints sur HF
- Phase 2a (attention) : <1s
- Phase 2b (embedding) : ~3.2h → checkpoints sur HF
- Phase 3 (joint) : ~41h → checkpoints sur HF toutes les 5000 steps

### Étape 8 : Assemblage final

Quand le training EDT est fini :
```bash
python3 scripts/transfer_to_cte.py
python3 scripts/assemble_fractus.py
```

---

## CE QUE JE DOIS CODER AVANT LA PROCHAINE SESSION

1. **Modifier `edt_pipeline.py` pour lire les shards .npy directement**
   - Au lieu de `torch.load('corpus.pt')`, faire `load_shards('data/fractus_1b_shards/')`
   - Échantillonne aléatoirement dans les shards pendant le training
   - Pas de fusion, pas de fichier unique

2. **Tester le code de lecture de shards en local**
   - Créer 3-4 petits shards factices
   - Vérifier que le training peut les lire
   - Vérifier que l'échantillonnage marche

3. **Tester EDT Phase 1 avec VRAIS hidden states**
   - Utiliser l'embedding pour générer des hidden states à partir du corpus local
   - Pas de données synthétiques

---

## COÛT ESTIMÉ PROCHAINE SESSION

| Item | Temps | Coût |
|------|-------|------|
| Setup | 10 min | $0.07 |
| Corpus build | 5-7h | $2-3 |
| GitHub dataset | 30 min | $0.20 |
| EDT Phase 1 | 1.2h | $0.50 |
| EDT Phase 2a | <1s | $0.00 |
| EDT Phase 2b | 3.2h | $1.30 |
| EDT Phase 3 | 41h | $16.40 |
| **Total** | **~50h** | **~$20-25** |

---

## CHECKLIST FINALE (à cocher avant de lancer)

- [ ] Tous les scripts testés en local (syntax + logique)
- [ ] `edt_pipeline.py` lit les shards directement (pas de fichier unique)
- [ ] Phase 1 utilise de vrais hidden states (pas synthétiques)
- [ ] HF_TOKEN configuré pour auto-push
- [ ] PGSU activé (--pgsu 4)
- [ ] 8-bit optimizer activé (--optim-8bit)
- [ ] Aux-loss clamp en place
- [ ] Corpus > 15B tokens vérifié
- [ ] Disk space > 200GB libre
- [ ] SSH testé avant de lancer quoi que ce soit
