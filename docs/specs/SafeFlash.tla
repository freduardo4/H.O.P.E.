--------------------------- MODULE SafeFlash ---------------------------
EXTENDS Integers, Sequences, TLC

(* 
  SafeFlash State Machine Specification
  Purpose: Formally verify that an ECU cannot be left in a non-recoverable 
  state without a backup, and that the flash process is transactional.
*)

VARIABLES 
    ecu_state,      \* Current ECU state: "Default", "Programming", "Bricked"
    flash_progress, \* Integer 0..100
    backup_status,  \* "None", "Local", "Cloud"
    last_error      \* Last recorded error

Vars == <<ecu_state, flash_progress, backup_status, last_error>>

TypeOK == 
    /\ ecu_state \in {"Default", "Programming", "Bricked"}
    /\ flash_progress \in 0..100
    /\ backup_status \in {"None", "Local", "Cloud"}

Init == 
    /\ ecu_state = "Default"
    /\ flash_progress = 0
    /\ backup_status = "None"
    /\ last_error = "None"

(* Atomic Transitions *)

CreateBackup ==
    /\ ecu_state = "Default"
    /\ backup_status = "None"
    /\ backup_status' = "Local"
    /\ UNCHANGED <<ecu_state, flash_progress, last_error>>

EnterProgramming ==
    /\ ecu_state = "Default"
    /\ backup_status \in {"Local", "Cloud"}
    /\ ecu_state' = "Programming"
    /\ UNCHANGED <<flash_progress, backup_status, last_error>>

FlashFailure ==
    /\ ecu_state = "Programming"
    /\ flash_progress < 100
    /\ ecu_state' = "Bricked"
    /\ last_error' = "ConnLoss"
    /\ UNCHANGED <<flash_progress, backup_status>>

RecoverySuccess ==
    /\ ecu_state = "Bricked"
    /\ backup_status \in {"Local", "Cloud"}
    /\ ecu_state' = "Default"
    /\ flash_progress' = 0
    /\ last_error' = "None"
    /\ UNCHANGED <<backup_status>>

Next == 
    \/ CreateBackup
    \/ EnterProgramming
    \/ FlashFailure
    \/ RecoverySuccess

(* Safety Properties *)

BackupRequiredForProg == 
    ecu_state = "Programming" => backup_status /= "None"

Recoverability ==
    ecu_state = "Bricked" => backup_status /= "None"

=============================================================================
