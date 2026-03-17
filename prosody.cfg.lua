-- Prosody XMPP config for local fraud MAS development

admins = {}

modules_enabled = {
    "roster"; "saslauth"; "tls"; "dialback"; "disco";
    "carbons"; "pep"; "private"; "blocklist"; "vcard4";
    "version"; "uptime"; "time"; "ping"; "register"; "admin_adhoc";
}

allow_registration = true
authentication     = "internal_plain"

-- Allow plain auth without TLS for local development
c2s_require_encryption       = false
allow_unencrypted_plain_auth = true

log = { info = "*stdout"; error = "*stderr" }

VirtualHost "localhost"

-- Auto-create agent accounts on first connect
registration_whitelist_only = false
