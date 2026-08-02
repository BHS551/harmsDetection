# IAM para el acceso de Claude Code a la infra de AWS

Este directorio contiene la política que otorga a la sesión de Claude Code acceso
para modificar la infraestructura de AWS, siguiendo la **Opción A** (inyección de
credenciales vía secretos del entorno de Claude Code on the web).

## Archivos

- `claude-code-broad-policy.json` — Política **A2 (recomendada)**: acceso amplio a
  los servicios elegidos (cómputo, almacenamiento, base de datos, red, IAM,
  CloudFormation) en **todas las regiones**, con `Deny` explícitos que impiden
  desmantelar la seguridad/control de la cuenta o auto-modificar el propio usuario.

## Alternativa A1 (máxima simpleza, máximo riesgo)

En lugar de la política de arriba, adjuntar la política gestionada de AWS
`AdministratorAccess` al usuario. Da acceso total sin barandas. Solo si se acepta
el riesgo.

## Pasos (resumen)

1. En **IAM → Users → Create user**, crear `claude-code-infra` (acceso programático).
2. Adjuntar la política:
   - A2: crear una *customer managed policy* con el contenido de
     `claude-code-broad-policy.json` y adjuntarla al usuario, **o**
   - A1: adjuntar directamente `AdministratorAccess`.
3. Generar el **Access key ID** y **Secret access key**.
4. Cargarlos como **secretos del entorno** en Claude Code on the web:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION` (ej. `us-east-1` como default para la CLI; el acceso no está
     limitado a esa región)
   - `AWS_DEFAULT_REGION`
5. Reiniciar la sesión para que el proxy inyecte las credenciales reales.
6. Verificación: `aws sts get-caller-identity` debe devolver el ARN de
   `claude-code-infra`.

## Notas de seguridad

- Las credenciales **nunca** deben pegarse en el chat; solo en el panel de secretos.
- El entorno es efímero: las credenciales no persisten entre sesiones, se reinyectan
  desde los secretos del entorno.
- Para endurecer más adelante: rotar la llave periódicamente, o migrar a credenciales
  temporales STS / rol asumible en vez de una llave de usuario de larga duración.
