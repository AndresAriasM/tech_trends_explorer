"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Plus, Trash2, FolderOpen, ChevronRight } from "lucide-react";
import { getCategories, createCategory, deleteCategory } from "@/lib/api";

export default function CategoriesPage() {
  const queryClient = useQueryClient();
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");
  const [error, setError] = useState("");

  const { data: categories = [], isLoading } = useQuery({
    queryKey: ["categories"],
    queryFn: getCategories,
  });

  const createMutation = useMutation({
    mutationFn: createCategory,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["categories"] });
      setNewName("");
      setNewDesc("");
    },
    onError: () => setError("Error al crear la categoría"),
  });

  const deleteMutation = useMutation({
    mutationFn: deleteCategory,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["categories"] }),
  });

  function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!newName.trim()) return;
    setError("");
    createMutation.mutate({ name: newName, description: newDesc });
  }

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Categorías</h1>
        <p className="text-gray-400 text-sm mt-1">
          Organiza tus análisis en grupos temáticos.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Create form */}
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
          <h2 className="font-semibold text-white text-sm mb-4">Nueva categoría</h2>
          <form onSubmit={handleCreate} className="space-y-3">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Nombre de la categoría"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
            />
            <textarea
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="Descripción (opcional)"
              rows={2}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500 resize-none"
            />
            {error && <p className="text-xs text-red-400">{error}</p>}
            <button
              type="submit"
              disabled={createMutation.isPending || !newName.trim()}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-900 disabled:cursor-not-allowed text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors"
            >
              <Plus className="w-4 h-4" />
              {createMutation.isPending ? "Creando..." : "Crear categoría"}
            </button>
          </form>
        </div>

        {/* Category list */}
        <div className="space-y-2">
          {isLoading && (
            <p className="text-sm text-gray-500 text-center py-8">Cargando...</p>
          )}
          {!isLoading && categories.length === 0 && (
            <div className="bg-gray-900 border border-dashed border-gray-700 rounded-xl p-8 text-center">
              <p className="text-gray-600 text-sm">No hay categorías aún.</p>
            </div>
          )}
          {categories.map((cat) => (
            <div
              key={cat.category_id}
              className="flex items-start gap-3 bg-gray-900 border border-gray-800 rounded-xl p-4"
            >
              <FolderOpen className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
              <div className="flex-1 min-w-0">
                <p className="font-medium text-white text-sm">{cat.name}</p>
                {cat.description && (
                  <p className="text-xs text-gray-500 mt-0.5 truncate">
                    {cat.description}
                  </p>
                )}
                {cat.technologies?.length > 0 && (
                  <div className="flex items-center gap-1 mt-1 text-xs text-gray-600">
                    <ChevronRight className="w-3 h-3" />
                    {cat.technologies.length} tecnología
                    {cat.technologies.length !== 1 ? "s" : ""}
                  </div>
                )}
              </div>
              {cat.category_id !== "default" && (
                <button
                  onClick={() => deleteMutation.mutate(cat.category_id)}
                  disabled={deleteMutation.isPending}
                  className="text-gray-600 hover:text-red-400 transition-colors disabled:opacity-40"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
