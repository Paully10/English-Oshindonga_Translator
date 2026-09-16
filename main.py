from config import TOKENIZER_FILE,CHECKPOINT_FILE
import subprocess,sys
if __name__=='__main__':
 if not TOKENIZER_FILE.exists(): subprocess.run([sys.executable,'tokenizer/tokenizer.py'],check=True)
 if not CHECKPOINT_FILE.exists(): print('No trained checkpoint. Run: python training/train.py'); raise SystemExit
 subprocess.run([sys.executable,'inference/translate.py'],check=True)
