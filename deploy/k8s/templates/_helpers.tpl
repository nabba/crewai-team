{{/* Common labels and helpers. */}}

{{- define "botarmy.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "botarmy.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "botarmy.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "botarmy.labels" -}}
app.kubernetes.io/name: {{ include "botarmy.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end -}}

{{- define "botarmy.gatewayImage" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
