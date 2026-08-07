import { Navigate, Route, Routes, Link } from "react-router-dom"
import { useSession, useLogout } from "@/hooks/useSession"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import SimulationPage from "@/pages/SimulationPage"
import AboutPage from "@/pages/AboutPage"
import LoginPage from "@/pages/LoginPage"

function RequireAuth({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useSession()

  if (isLoading) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <Skeleton className="h-10 w-64" />
      </div>
    )
  }
  if (!data?.authenticated) {
    return <Navigate to="/login" replace />
  }
  return <>{children}</>
}

function NavBar() {
  const { data } = useSession()
  const logout = useLogout()

  if (!data?.authenticated) return null

  return (
    <nav className="border-b">
      <div className="mx-auto max-w-5xl px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <span className="font-semibold">SPRINGS ABM</span>
          <Link to="/" className="text-sm text-muted-foreground hover:text-foreground">
            Simulation
          </Link>
          <Link to="/about" className="text-sm text-muted-foreground hover:text-foreground">
            About the model
          </Link>
        </div>
        <Button variant="ghost" size="sm" onClick={() => logout.mutate()}>
          Log out
        </Button>
      </div>
    </nav>
  )
}

function App() {
  return (
    <div className="min-h-svh">
      <NavBar />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route
          path="/"
          element={
            <RequireAuth>
              <SimulationPage />
            </RequireAuth>
          }
        />
        <Route
          path="/runs/:jobId"
          element={
            <RequireAuth>
              <SimulationPage />
            </RequireAuth>
          }
        />
        <Route
          path="/about"
          element={
            <RequireAuth>
              <AboutPage />
            </RequireAuth>
          }
        />
      </Routes>
    </div>
  )
}

export default App
