# Windows Scripts Guide

This directory contains both **batch files** (.bat) and **PowerShell scripts** (.ps1) for Windows users.

## 🎯 Which Scripts to Use?

### PowerShell (Recommended) ✅
- **Better**: Colored output, error handling, modern syntax
- **Requires**: PowerShell 5.1+ (pre-installed on Windows 10/11)
- **How to run**: `.\scripts\script_name.ps1`

### Batch Files (Legacy) 
- **Good for**: Older Windows systems, simple automation
- **Works on**: Any Windows version with cmd.exe
- **How to run**: `scripts\script_name.bat`

---

## 🚀 Quick Start Commands

### Training

**PowerShell:**
```powershell
.\scripts\train.ps1
```

**Batch:**
```cmd
scripts\train.bat
```

### Serving (Inference API)

**PowerShell:**
```powershell
.\scripts\serve.ps1
```

**Batch:**
```cmd
scripts\serve.bat
```

### Export (Backup)

**PowerShell:**
```powershell
# Auto-generated name
.\scripts\docker_export.ps1

# Custom name
.\scripts\docker_export.ps1 my_backup
```

**Batch:**
```cmd
REM Auto-generated name
scripts\docker_export.bat

REM Custom name
scripts\docker_export.bat my_backup
```

### Import (Restore)

**PowerShell:**
```powershell
# Preview (dry-run)
.\scripts\docker_import.ps1 my_backup.zip --dry-run

# Import
.\scripts\docker_import.ps1 my_backup.zip

# Overwrite existing data
.\scripts\docker_import.ps1 my_backup.zip --overwrite
```

**Batch:**
```cmd
REM Preview (dry-run)
scripts\docker_import.bat my_backup.zip --dry-run

REM Import
scripts\docker_import.bat my_backup.zip

REM Overwrite existing data
scripts\docker_import.bat my_backup.zip --overwrite
```

---

## 🛠️ PowerShell Execution Policy

If you get an error like "running scripts is disabled", you need to enable script execution:

### Option 1: Enable for Current Session (Temporary)
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
```

### Option 2: Enable for Current User (Recommended)
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Option 3: Run Without Changing Policy
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train.ps1
```

---

## 📋 Full Command Reference

### Training Scripts

| Task | PowerShell | Batch |
|------|------------|-------|
| Train model | `.\scripts\train.ps1` | `scripts\train.bat` |

### Serving Scripts

| Task | PowerShell | Batch |
|------|------------|-------|
| Start API | `.\scripts\serve.ps1` | `scripts\serve.bat` |

### Export Scripts

| Task | PowerShell | Batch |
|------|------------|-------|
| Auto backup | `.\scripts\docker_export.ps1` | `scripts\docker_export.bat` |
| Named backup | `.\scripts\docker_export.ps1 my_backup` | `scripts\docker_export.bat my_backup` |

### Import Scripts

| Task | PowerShell | Batch |
|------|------------|-------|
| Dry run | `.\scripts\docker_import.ps1 file.zip --dry-run` | `scripts\docker_import.bat file.zip --dry-run` |
| Import | `.\scripts\docker_import.ps1 file.zip` | `scripts\docker_import.bat file.zip` |
| Overwrite | `.\scripts\docker_import.ps1 file.zip --overwrite` | `scripts\docker_import.bat file.zip --overwrite` |

---

## 🐛 Troubleshooting

### "Docker not found"
Make sure Docker Desktop is installed and running. Check with:
```cmd
docker --version
docker-compose --version
```

### "Script not recognized"
Make sure you're running from the project root directory:
```powershell
cd C:\path\to\MLOps
.\scripts\train.ps1
```

### Batch file doesn't show colors
This is normal - batch files don't support colored output. Use PowerShell for a better experience.

### PowerShell script won't run
See the "PowerShell Execution Policy" section above.

### Path issues with backups
Windows uses backslashes (`\`) for paths, but Docker uses forward slashes (`/`). The scripts handle this automatically.

---

## 💡 Tips for Windows Users

### 1. Use PowerShell ISE or VS Code
Better syntax highlighting and debugging:
```powershell
# Open in PowerShell ISE
powershell_ise.exe .\scripts\train.ps1

# Open in VS Code
code .\scripts\train.ps1
```

### 2. Create Desktop Shortcuts
Right-click on a `.bat` or `.ps1` file → Send to → Desktop (create shortcut)

### 3. Add to Windows Terminal
Windows Terminal can run PowerShell, CMD, and WSL. Create custom profiles for common tasks.

### 4. Use Task Scheduler for Automation
Schedule regular backups:
```powershell
# Example: Daily backup at 2 AM
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\path\to\MLOps\scripts\docker_export.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "MLflow Daily Backup"
```

---

## 📁 File Locations

All scripts follow the same structure as Linux:

- **Backups**: `.\backups\`
- **Dataset**: `.\dataset\`
- **Experiments**: `.\experiments\`
- **Models**: `.\models\`

---

## 🔄 Equivalent Commands

| Linux/Mac | Windows (PowerShell) | Windows (CMD) |
|-----------|----------------------|---------------|
| `./script.sh` | `.\script.ps1` | `script.bat` |
| `ls -la` | `Get-ChildItem` or `dir` | `dir` |
| `chmod +x script.sh` | Not needed | Not needed |
| `cat file.txt` | `Get-Content file.txt` | `type file.txt` |
| `rm file.txt` | `Remove-Item file.txt` | `del file.txt` |

---

## 📚 Additional Resources

- [PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/)
- [Windows Terminal](https://aka.ms/terminal)
- Main Documentation: `DOCKER_BACKUP.md`, `BACKUP_README.md`

---

## ✅ Quick Checklist

Before running scripts, ensure:
- [ ] Docker Desktop is installed and running
- [ ] PowerShell execution policy is set (for .ps1 files)
- [ ] You're in the project root directory
- [ ] MLflow services are running (`docker-compose up -d`)
- [ ] Required directories exist (`backups`, `dataset`, etc.)

---

**Need help?** See `DOCKER_BACKUP.md` for detailed documentation.
