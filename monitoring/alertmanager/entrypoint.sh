#!/bin/sh
# Starts Alertmanager with alertmanager.yml's ${...} placeholders filled in
# from the environment (.env.alertmanager), so no address or login is kept
# in git. The SMTP password isn't substituted: Alertmanager reads it from
# secrets/smtp_password itself.

set -eu

for var in SMTP_SMARTHOST SMTP_FROM SMTP_USERNAME SMTP_REQUIRE_TLS ALERT_EMAIL_TO; do
    eval "value=\${$var:-}"
    if [ -z "$value" ]; then
        echo "alertmanager: $var is not set (see .env.alertmanager.example)" >&2
        exit 1
    fi
done

if [ ! -s /etc/alertmanager/secrets/smtp_password ]; then
    echo "alertmanager: monitoring/alertmanager/secrets/smtp_password is missing" >&2
    exit 1
fi

# Escapes characters that mean something in a sed replacement.
escape() {
    printf '%s' "$1" | sed -e 's/[|&\\]/\\&/g'
}

sed \
    -e "s|\${SMTP_SMARTHOST}|$(escape "$SMTP_SMARTHOST")|g" \
    -e "s|\${SMTP_FROM}|$(escape "$SMTP_FROM")|g" \
    -e "s|\${SMTP_USERNAME}|$(escape "$SMTP_USERNAME")|g" \
    -e "s|\${SMTP_REQUIRE_TLS}|$(escape "$SMTP_REQUIRE_TLS")|g" \
    -e "s|\${ALERT_EMAIL_TO}|$(escape "$ALERT_EMAIL_TO")|g" \
    /etc/alertmanager/alertmanager.yml > /tmp/alertmanager.yml

exec /bin/alertmanager \
    --config.file=/tmp/alertmanager.yml \
    --storage.path=/alertmanager \
    "$@"
