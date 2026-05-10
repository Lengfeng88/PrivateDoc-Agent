// Azure Container Apps — FastAPI backend
resource api 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'privatedoc-api'
  location: location
  properties: {
    configuration: {
      ingress: { external: true, targetPort: 8000 }
      secrets: [
        { name: 'azure-oai-key', value: azureOaiKey }
        { name: 'langsmith-key', value: langsmithKey }
      ]
    }
    template: {
      containers: [{
        name: 'api'
        image: 'ghcr.io/${githubUser}/privatedoc-api:latest'
        env: [
          { name: 'AZURE_OAI_KEY',    secretRef: 'azure-oai-key' }
          { name: 'LANGSMITH_API_KEY', secretRef: 'langsmith-key' }
          { name: 'LOCAL_LLM_URL',    value: 'http://localhost:8080' }
          { name: 'CLOUD_FALLBACK',   value: 'true' }
        ]
        resources: { cpu: '1.0', memory: '2Gi' }
      }]
      scale: { minReplicas: 0, maxReplicas: 3 }  // scale-to-zero = $0 when idle
    }
  }
}