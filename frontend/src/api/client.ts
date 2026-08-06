import type {
  ParametersResponse,
  RunDetail,
  RunListResponse,
  ScenarioFormValues,
  SessionResponse,
  SubmitRunResponse,
} from "./types"

export class ApiError extends Error {
  status: number
  detail: unknown

  constructor(message: string, status: number, detail?: unknown) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  })

  if (resp.status === 401) {
    // A 401 from anywhere but the login attempt itself means the session
    // expired mid-use - bounce to /login. The login endpoint's own 401
    // ("wrong password") is an expected, inline-handled failure, not a
    // session expiry, so it's excluded here and left to the caller.
    const isLoginAttempt = path === "/login"
    if (!isLoginAttempt && window.location.pathname !== "/login") {
      window.location.assign("/login")
    }
  }

  if (!resp.ok) {
    const body = await resp.json().catch(() => null)
    const message = typeof body?.detail === "string" ? body.detail : resp.statusText
    throw new ApiError(message, resp.status, body?.detail)
  }

  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

export const api = {
  getSession: () => request<SessionResponse>("/session"),
  login: (password: string) =>
    request<SessionResponse>("/login", { method: "POST", body: JSON.stringify({ password }) }),
  logout: () => request<SessionResponse>("/logout", { method: "POST" }),

  getParameters: () => request<ParametersResponse>("/parameters"),

  submitScenario: (values: ScenarioFormValues) =>
    request<SubmitRunResponse>("/scenario/run", { method: "POST", body: JSON.stringify(values) }),

  getRun: (jobId: string) => request<RunDetail>(`/runs/${jobId}`),
  listRuns: (limit = 20) => request<RunListResponse>(`/runs?limit=${limit}`),
}
