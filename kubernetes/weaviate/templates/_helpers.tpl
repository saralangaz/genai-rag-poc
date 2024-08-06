{{/*
Generate a name for the deployment
*/}}
{{- define "weaviate.fullname" -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Generate the name of the app
*/}}
{{- define "weaviate.name" -}}
{{- .Chart.Name -}}
{{- end -}}
